"""AutoVid command-line interface.

Heavy modules (numpy/PIL/librosa chains) are imported inside the command
bodies, not at module level: ``autovid status`` / ``edls`` / ``--help`` must
not pay for them on every invocation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import db, edl as edl_mod
from .config import Settings, get_settings
from .ingest import ingest_folder

app = typer.Typer(
    no_args_is_help=True,
    help="AutoVid — AI-powered automatic video editor.",
    add_completion=False,
)
console = Console()


@app.callback()
def _root() -> None:
    """AutoVid — AI-powered automatic video editor."""


@app.command()
def ingest(
    folder: Path = typer.Argument(..., exists=True, file_okay=False, help="Folder of raw videos."),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", help="Scan subfolders."),
    data_dir: Path | None = typer.Option(None, help="Catalog dir (default AUTOVID_DATA_DIR)."),
) -> None:
    """Discover videos, dedupe by content, store metadata, generate thumbnails."""
    settings: Settings = get_settings()
    if data_dir is not None:
        settings = settings.model_copy(update={"data_dir": data_dir})
    conn = db.connect(settings.db_path)

    with console.status("Ingesting…") as status:
        report = ingest_folder(
            conn, settings, folder, recursive=recursive,
            progress=lambda p, i, n: status.update(f"[{i}/{n}] {Path(p).name}"),
        )

    table = Table(title="Ingest summary", show_lines=False)
    table.add_column("outcome")
    table.add_column("count", justify="right")
    table.add_column("files", overflow="fold")
    for label, items in (
        ("added", report.added),
        ("duplicates", report.duplicates),
        ("unchanged (skipped)", report.skipped),
        ("failed", report.failed),
    ):
        table.add_row(label, str(len(items)), "\n".join(items[:5]) or "—")
    console.print(table)


@app.command()
def sample(
    asset_id: str | None = typer.Argument(None, help="Specific asset id (a_…)."),
    all: bool = typer.Option(False, "--all", help="Sample every ingested asset."),
    data_dir: Path | None = typer.Option(None, help="Catalog dir (default AUTOVID_DATA_DIR)."),
) -> None:
    """Baseline sampling: scene boundaries + grid frames + motion metrics."""
    from .sampling import pending_assets, sample_asset

    settings: Settings = get_settings()
    if data_dir is not None:
        settings = settings.model_copy(update={"data_dir": data_dir})
    conn = db.connect(settings.db_path)

    targets = [row["id"] for row in pending_assets(conn)] if (
        all or asset_id is None
    ) else [asset_id]
    if asset_id is not None and all:
        targets = [asset_id]
    if not targets:
        console.print("[yellow]Nothing to sample (no ingested assets).[/yellow]")
        raise typer.Exit()

    with console.status("Sampling…") as status:
        for i, aid in enumerate(targets, 1):
            status.update(f"[{i}/{len(targets)}] {aid}")
            try:
                result = sample_asset(conn, settings, aid)
            except Exception as e:  # noqa: BLE001
                console.print(f"[red]{aid}: {e}[/red]")
                continue
            console.print(
                f"[green]{aid}[/green]: {result.scene_changes} scene changes, "
                f"{result.grid_frames} grid frames"
            )


@app.command()
def analyze(
    asset_id: str | None = typer.Argument(None, help="Specific asset id (a_…)."),
    all: bool = typer.Option(False, "--all", help="Analyze every sampled asset."),
    data_dir: Path | None = typer.Option(None, help="Catalog dir (default AUTOVID_DATA_DIR)."),
) -> None:
    """Build the temporal map: D1-guided window sampling + cloud VLM descriptions."""
    from .analyze import analyze_asset
    from .decisions import DecisionClient
    from .sampling import pending_assets
    from .vlm import VLMClient

    settings: Settings = get_settings()
    if data_dir is not None:
        settings = settings.model_copy(update={"data_dir": data_dir})
    conn = db.connect(settings.db_path)

    targets = [row["id"] for row in pending_assets(conn, "sampled")] if (
        all or asset_id is None
    ) else [asset_id]
    if not targets:
        console.print("[yellow]Nothing to analyze (no sampled assets).[/yellow]")
        raise typer.Exit()

    vlm = VLMClient(settings)
    if not vlm.configured:
        console.print(
            "[yellow]No AUTOVID_VLM_API_KEY set — windows will be marked uncertain "
            "and flagged for review (decision layer still runs).[/yellow]"
        )
    decisions = DecisionClient(settings, conn)

    with console.status("Analyzing…") as status:
        for i, aid in enumerate(targets, 1):
            status.update(f"[{i}/{len(targets)}] {aid}")
            try:
                result = analyze_asset(conn, settings, aid, vlm=vlm, decisions=decisions)
            except Exception as e:  # noqa: BLE001
                console.print(f"[red]{aid}: {e}[/red]")
                continue
            console.print(
                f"[green]{aid}[/green]: {result.windows} windows, "
                f"{result.dense_frames} dense frames, {result.flagged} flagged, "
                f"{result.prompt_tokens + result.completion_tokens} tokens"
            )
            for note in result.notes:
                console.print(f"  [dim]{note}[/dim]")


@app.command()
def cut(
    asset_id: str | None = typer.Argument(None, help="Specific asset id (a_…)."),
    all: bool = typer.Option(False, "--all", help="Cut every analyzed asset."),
    data_dir: Path | None = typer.Option(None, help="Catalog dir (default AUTOVID_DATA_DIR)."),
) -> None:
    """Verify boundaries, accept candidate segments, extract clips."""
    from .decisions import DecisionClient
    from .refine import cut_asset
    from .sampling import pending_assets

    settings: Settings = get_settings()
    if data_dir is not None:
        settings = settings.model_copy(update={"data_dir": data_dir})
    conn = db.connect(settings.db_path)

    targets = [row["id"] for row in pending_assets(conn, "analyzed")] if (
        all or asset_id is None
    ) else [asset_id]
    if not targets:
        console.print("[yellow]Nothing to cut (no analyzed assets).[/yellow]")
        raise typer.Exit()

    decisions = DecisionClient(settings, conn)
    with console.status("Cutting…") as status:
        for i, aid in enumerate(targets, 1):
            status.update(f"[{i}/{len(targets)}] {aid}")
            try:
                result = cut_asset(conn, settings, aid, decisions=decisions)
            except Exception as e:  # noqa: BLE001
                console.print(f"[red]{aid}: {e}[/red]")
                continue
            console.print(
                f"[green]{aid}[/green]: {result.windows} windows → {result.scenes} scenes "
                f"({result.boundaries_verified} boundaries verified, "
                f"{result.boundaries_refined} refined), "
                f"segments {result.segments_accepted} accepted / "
                f"{result.segments_flagged} flagged / {result.segments_rejected} rejected, "
                f"{result.clips_extracted} clips"
            )


@app.command()
def organize(
    asset_id: str | None = typer.Argument(None, help="Specific asset id (a_…)."),
    all: bool = typer.Option(False, "--all", help="Organize every cut asset."),
    data_dir: Path | None = typer.Option(None, help="Catalog dir (default AUTOVID_DATA_DIR)."),
) -> None:
    """Tag segments, flag quality, suggest recoverable filenames."""
    from .organize import organize_asset
    from .sampling import pending_assets

    settings: Settings = get_settings()
    if data_dir is not None:
        settings = settings.model_copy(update={"data_dir": data_dir})
    conn = db.connect(settings.db_path)

    targets = [row["id"] for row in pending_assets(conn, "cut")] if (
        all or asset_id is None
    ) else [asset_id]
    if not targets:
        console.print("[yellow]Nothing to organize (no cut assets).[/yellow]")
        raise typer.Exit()

    with console.status("Organizing…") as status:
        for i, aid in enumerate(targets, 1):
            status.update(f"[{i}/{len(targets)}] {aid}")
            try:
                result = organize_asset(conn, settings, aid)
            except Exception as e:  # noqa: BLE001
                console.print(f"[red]{aid}: {e}[/red]")
                continue
            console.print(f"[green]{aid}[/green]: {result.segments_tagged} segments tagged")


@app.command()
def search(
    query: str = typer.Argument(..., help="Text to find in descriptions, tags, names."),
    data_dir: Path | None = typer.Option(None, help="Catalog dir (default AUTOVID_DATA_DIR)."),
) -> None:
    """Search the segment catalog."""
    from .organize import search as search_segments

    settings: Settings = get_settings()
    if data_dir is not None:
        settings = settings.model_copy(update={"data_dir": data_dir})
    conn = db.connect(settings.db_path)

    rows = search_segments(conn, query)
    if not rows:
        console.print("[yellow]No matches.[/yellow]")
        raise typer.Exit()
    table = Table(title=f"Search: {query!r} ({len(rows)} hits)")
    table.add_column("seg", justify="right")
    table.add_column("span")
    table.add_column("verdict")
    table.add_column("score", justify="right")
    table.add_column("description", overflow="fold")
    table.add_column("tags", overflow="fold")
    table.add_column("clip")
    for r in rows:
        tags = ", ".join(json.loads(r["tags_json"] or "[]"))
        table.add_row(
            str(r["id"]), f"{r['start_s']:.1f}–{r['end_s']:.1f}s", r["verdict"] or "—",
            f"{r['highlight_score']:.2f}" if r["highlight_score"] is not None else "—",
            r["description"], tags,
            "yes" if r["clip_path"] else "no",
        )
    console.print(table)


@app.command()
def rename(
    segment_id: int = typer.Argument(None, help="Segment whose clip to rename."),
    undo: int | None = typer.Option(None, "--undo", help="Rename-journal id to revert."),
    data_dir: Path | None = typer.Option(None, help="Catalog dir (default AUTOVID_DATA_DIR)."),
) -> None:
    """Rename an extracted clip to its suggested name (journal-backed, reversible)."""
    from .organize import apply_rename, undo_rename

    settings: Settings = get_settings()
    if data_dir is not None:
        settings = settings.model_copy(update={"data_dir": data_dir})
    conn = db.connect(settings.db_path)
    if undo is not None:
        undo_rename(conn, undo)
        console.print(f"[green]Reverted rename #{undo}.[/green]")
        return
    if segment_id is None:
        console.print("[red]Provide a SEGMENT_ID or --undo RENAME_ID.[/red]")
        raise typer.Exit(1)
    rename_id = apply_rename(conn, segment_id)
    row = conn.execute(
        "SELECT new_path FROM renames WHERE id = ?", (rename_id,)
    ).fetchone()
    console.print(f"[green]Renamed (journal #{rename_id}):[/green] {row['new_path']}")


@app.command()
def storyboard(
    objective: str = typer.Argument(..., help="Editing objective, e.g. '60s hike recap'."),
    duration: float = typer.Option(30.0, help="Target duration in seconds."),
    style: str = typer.Option(
        "cinematic", help="fast_paced | social | cinematic | informative."
    ),
    name: str = typer.Option("rough-cut", help="EDL name."),
    data_dir: Path | None = typer.Option(None, help="Catalog dir (default AUTOVID_DATA_DIR)."),
) -> None:
    """Select accepted segments into an ordered, editable EDL."""
    from .storyboard import select_clips

    settings: Settings = get_settings()
    if data_dir is not None:
        settings = settings.model_copy(update={"data_dir": data_dir})
    conn = db.connect(settings.db_path)

    try:
        board = select_clips(
            conn, objective=objective, target_duration_s=duration,
            style=style, name=name,
        )
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    console.print(
        f"[green]EDL {board.id}[/green]: {len(board.items)} items, "
        f"{board.duration_s:.1f}s (target {duration:.1f}s) — edit via the web UI "
        f"or the edls table, then render."
    )
    for i, item in enumerate(board.items, 1):
        console.print(
            f"  {i:2d}. seg {item.clip_id} {item.source_start:.1f}–{item.source_end:.1f}s "
            f"[{item.transition_in}→{item.transition_out}]"
        )


@app.command()
def agent(
    objective: str = typer.Argument(..., help="Editing objective, e.g. '60s hike recap'."),
    duration: float = typer.Option(30.0, help="Target duration in seconds."),
    style: str = typer.Option(
        "cinematic", help="fast_paced | social | cinematic | informative."),
    name: str = typer.Option("storyboard", help="Storyboard name."),
    plan_only: bool = typer.Option(False, "--plan-only", help="Skip compiling to an EDL."),
    data_dir: Path | None = typer.Option(None, help="Catalog dir (default AUTOVID_DATA_DIR)."),
) -> None:
    """Storyboard Agent: script + storyboard from the catalog.

    Uses the text LLM when AUTOVID_VLM_API_KEY is set (catalog-gated — it can
    only reference real segments); otherwise, or on invalid drafts, falls
    back to the deterministic planner.
    """
    from . import storyboard_doc as doc_mod
    from .agent import generate_with_fallback

    settings: Settings = get_settings()
    if data_dir is not None:
        settings = settings.model_copy(update={"data_dir": data_dir})
    conn = db.connect(settings.db_path)

    try:
        board_id, origin = generate_with_fallback(
            conn, settings, objective=objective, target_duration_s=duration,
            style=style, name=name,
        )
    except Exception as e:  # noqa: BLE001 — surfaced to the user
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    row = doc_mod.get_board(conn, board_id)
    document, version = doc_mod.load_document(conn, board_id)
    console.print(
        f"[green]Storyboard {board_id} v{version}[/green] (via {origin}): "
        f"{len(document.scenes)} scenes, {document.duration_s:.1f}s "
        f"(target {duration:.1f}s)")
    for scene in document.scenes:
        shot = scene.shots[0] if scene.shots else None
        span = f"seg {shot.segment_id} {shot.source_start:.1f}-{shot.source_end:.1f}s" \
            if shot else "no shots"
        console.print(
            f"  {scene.id:>4} [{scene.narrative_role or '—'}] {scene.title}: {span}"
            + (" [stale]" if scene.stale else "")
            + (" [locked]" if scene.locked else ""))
    if document.missing_footage:
        console.print("[yellow]Missing footage the story asked for:[/yellow]")
        for m in document.missing_footage:
            console.print(f"  - {m}")
    if plan_only:
        return
    try:
        board = doc_mod.compile_to_edl(conn, board_id)
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Compile failed: {e}[/red]")
        raise typer.Exit(1)
    console.print(
        f"[green]EDL {board.id}[/green]: {len(board.items)} items, "
        f"{board.duration_s:.1f}s — render it or fine-tune in the web UI.")


@app.command()
def boards(
    data_dir: Path | None = typer.Option(None, help="Catalog dir (default AUTOVID_DATA_DIR)."),
) -> None:
    """List saved storyboards (the agent/planner planning artifacts)."""
    from . import storyboard_doc as doc_mod

    settings: Settings = get_settings()
    if data_dir is not None:
        settings = settings.model_copy(update={"data_dir": data_dir})
    conn = db.connect(settings.db_path)
    rows = doc_mod.list_boards(conn)
    if not rows:
        console.print("[yellow]No storyboards yet — run `autovid agent`.[/yellow]")
        raise typer.Exit()
    table = Table(title="Storyboards")
    table.add_column("id")
    table.add_column("name")
    table.add_column("objective", overflow="fold")
    table.add_column("style")
    table.add_column("status")
    table.add_column("ver", justify="right")
    table.add_column("updated")
    for r in rows:
        table.add_row(r["id"], r["name"], r["objective"], r["style"],
                      r["status"], str(r["versions"]), r["updated_at"])
    console.print(table)


@app.command()
def edls(
    data_dir: Path | None = typer.Option(None, help="Catalog dir (default AUTOVID_DATA_DIR)."),
) -> None:
    """List saved EDLs."""
    settings: Settings = get_settings()
    if data_dir is not None:
        settings = settings.model_copy(update={"data_dir": data_dir})
    conn = db.connect(settings.db_path)
    rows = edl_mod.list_edls(conn)
    if not rows:
        console.print("[yellow]No EDLs yet.[/yellow]")
        raise typer.Exit()
    table = Table(title="EDLs")
    table.add_column("id")
    table.add_column("name")
    table.add_column("objective", overflow="fold")
    table.add_column("style")
    table.add_column("target (s)", justify="right")
    table.add_column("updated")
    for r in rows:
        table.add_row(r["id"], r["name"], r["objective"], r["style"],
                      f"{r['target_duration_s']:.0f}", r["updated_at"])
    console.print(table)


@app.command()
def render(
    edl_id: str = typer.Argument(..., help="EDL id (see `autovid edls`)."),
    data_dir: Path | None = typer.Option(None, help="Catalog dir (default AUTOVID_DATA_DIR)."),
) -> None:
    """Render a rough-cut preview MP4 from an EDL."""
    from .render import render_edl

    settings: Settings = get_settings()
    if data_dir is not None:
        settings = settings.model_copy(update={"data_dir": data_dir})
    conn = db.connect(settings.db_path)
    board = edl_mod.load(conn, edl_id)
    with console.status("Rendering…"):
        try:
            out = render_edl(conn, settings, board)
        except Exception as e:  # noqa: BLE001 — validation + render errors
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)
    console.print(f"[green]Rendered:[/green] {out} ({board.duration_s:.1f}s)")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind address."),
    port: int = typer.Option(8766, help="Bind port."),
    data_dir: Path | None = typer.Option(None, help="Catalog dir (default AUTOVID_DATA_DIR)."),
) -> None:
    """Review UI: catalog, review queue, EDL editor, render preview."""
    settings: Settings = get_settings()
    if data_dir is not None:
        settings = settings.model_copy(update={"data_dir": data_dir})
    console.print(f"Serving review UI on http://{host}:{port} (Ctrl+C to stop)")
    from .web import serve as serve_web

    serve_web(settings, host=host, port=port)


@app.command()
def bench(
    ground_truth: Path = typer.Argument(
        ..., help="JSON: {asset name or id: [true boundary seconds…]}"),
    name: str = typer.Option("bench", help="Report name (catalog/bench/<name>/)."),
    data_dir: Path | None = typer.Option(None, help="Catalog dir (default AUTOVID_DATA_DIR)."),
) -> None:
    """Compare fixed-interval baseline vs decision-guided processing."""
    from .bench import run_benchmark
    from .decisions import DecisionClient
    from .vlm import VLMClient

    settings: Settings = get_settings()
    if data_dir is not None:
        settings = settings.model_copy(update={"data_dir": data_dir})
    conn = db.connect(settings.db_path)

    truth = json.loads(Path(ground_truth).read_text(encoding="utf-8"))
    decisions = DecisionClient(settings, conn)
    report = run_benchmark(
        conn, settings, vlm=VLMClient(settings), decisions=decisions,
        ground_truth=truth, name=name,
    )
    md = Path(settings.data_dir) / "bench" / name / "report.md"
    console.print(md.read_text(encoding="utf-8"))
    console.print(f"[dim]Full report: {md.with_suffix('.json')}[/dim]")


@app.command()
def approve(
    segment_id: int = typer.Argument(..., help="Segment id from the review queue."),
    reject: bool = typer.Option(False, "--reject", help="Reject instead of approve."),
    data_dir: Path | None = typer.Option(None, help="Catalog dir (default AUTOVID_DATA_DIR)."),
) -> None:
    """Human review decision: make a flagged segment (in)eligible for storyboards."""
    settings: Settings = get_settings()
    if data_dir is not None:
        settings = settings.model_copy(update={"data_dir": data_dir})
    conn = db.connect(settings.db_path)
    status = "rejected" if reject else "approved"
    cur = conn.execute(
        "UPDATE segments SET review_status = ? WHERE id = ?", (status, segment_id)
    )
    conn.commit()
    if cur.rowcount == 0:
        console.print(f"[red]Unknown segment {segment_id}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]Segment {segment_id} → {status}[/green]")


@app.command()
def musicvideo(
    music: Path = typer.Argument(..., exists=True, help="Music/BGM file (any ffmpeg-readable audio)."),
    objective: str = typer.Option("", help="What the edit is for, e.g. 'event recap'."),
    mode: str = typer.Option("hybrid", help="beat | cinematic | hybrid cut placement."),
    style: str = typer.Option("cinematic", help="Pacing preset (beats per cut by section role)."),
    name: str = typer.Option("music-cut", help="EDL / render name."),
    render: bool = typer.Option(True, "--render/--plan-only", help="Render after planning."),
    data_dir: Path | None = typer.Option(None, help="Catalog dir (default AUTOVID_DATA_DIR)."),
) -> None:
    """Plan (and render) a music-driven edit from the scene catalog.

    Beats, tempo, energy and sections are measured from the music; clip
    selection and cut placement come from the catalog; the decision layer
    reviews each section's draft plan.
    """
    from .decisions import DecisionClient
    from .music import analyze_music
    from .planner import plan_music_video
    from .render import render_edl

    settings: Settings = get_settings()
    if data_dir is not None:
        settings = settings.model_copy(update={"data_dir": data_dir})
    conn = db.connect(settings.db_path)

    with console.status("Analyzing music…"):
        analysis = analyze_music(music)
    console.print(
        f"[green]{analysis.tempo_bpm:.0f} BPM[/green], {len(analysis.beats)} beats, "
        f"{len(analysis.sections)} sections, {analysis.duration_s:.1f}s"
    )

    decisions = DecisionClient(settings, conn)
    with console.status("Planning…"):
        board = plan_music_video(
            conn, settings, analysis, objective=objective, mode=mode,
            style=style, name=name, decisions=decisions,
        )
    if not board.items:
        console.print("[red]Planner produced no items — is the catalog populated?[/red]")
        raise typer.Exit(1)
    console.print(
        f"[green]EDL {board.id}[/green]: {len(board.items)} cuts over "
        f"{board.duration_s:.1f}s of music (mode {board.mode})"
    )
    for i, item in enumerate(board.items, 1):
        console.print(
            f"  {i:2d}. {item.timeline_start:5.1f}–{item.timeline_end:5.1f}s "
            f"← seg {item.clip_id} {item.source_start:.1f}–{item.source_end:.1f}s "
            f"[{item.beat_ref or '—'}]"
        )

    if not render:
        return
    with console.status("Rendering…"):
        out = render_edl(conn, settings, board)
    console.print(f"[green]Rendered:[/green] {out}")


@app.command("app")
def app_cmd(
    port: int = typer.Option(8766, help="Port for the app server."),
    browser: bool = typer.Option(False, "--browser", help="Open in a browser tab "
                                 "instead of a native window (no pywebview needed)."),
    data_dir: Path | None = typer.Option(None, help="Catalog dir (default AUTOVID_DATA_DIR)."),
    make_launcher: bool = typer.Option(False, "--make-launcher", help="Write a "
                                       "double-clickable AutoVid.bat to the Desktop and exit."),
) -> None:
    """Launch the AutoVid desktop app (settings, pipeline, review, music)."""
    settings: Settings = get_settings()
    if data_dir is not None:
        settings = settings.model_copy(update={"data_dir": data_dir})
    from .app import launch, write_launcher_script

    if make_launcher:
        console.print(f"[green]Launcher:[/green] {write_launcher_script(Path.home() / 'Desktop')}")
        return
    launch(settings, port=port, window=not browser)


@app.command()
def status(
    data_dir: Path | None = typer.Option(None, help="Catalog dir (default AUTOVID_DATA_DIR)."),
) -> None:
    """Catalog overview: assets, pipeline status, scene/frame counts."""
    settings: Settings = get_settings()
    if data_dir is not None:
        settings = settings.model_copy(update={"data_dir": data_dir})
    conn = db.connect(settings.db_path)

    rows = conn.execute(
        """SELECT a.id, a.original_name, a.status, a.duration_s, a.has_audio,
                  (SELECT COUNT(*) FROM scenes s WHERE s.asset_id = a.id) AS scenes,
                  (SELECT COUNT(*) FROM frames f WHERE f.asset_id = a.id) AS frames,
                  (SELECT COUNT(*) FROM segments g WHERE g.asset_id = a.id) AS segments
           FROM assets a ORDER BY a.created_at"""
    ).fetchall()
    table = Table(title=f"Catalog ({len(rows)} assets)")
    table.add_column("asset")
    table.add_column("name", overflow="fold")
    table.add_column("status")
    table.add_column("dur (s)", justify="right")
    table.add_column("audio", justify="center")
    table.add_column("scenes", justify="right")
    table.add_column("frames", justify="right")
    table.add_column("segments", justify="right")
    for r in rows:
        table.add_row(
            r["id"], r["original_name"], r["status"], f"{r['duration_s']:.1f}",
            "yes" if r["has_audio"] else "no",
            str(r["scenes"]), str(r["frames"]), str(r["segments"]),
        )
    console.print(table)


def _utf8_stdio() -> None:
    # Windows consoles default to cp1252, which can't print the arrows in our output.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    _utf8_stdio()
    app()


if __name__ == "__main__":
    main()
