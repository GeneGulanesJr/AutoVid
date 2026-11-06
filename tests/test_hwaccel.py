from autovid.ffmpeg_io.bins import hwaccel_args


def test_hwaccel_env(monkeypatch):
    monkeypatch.setenv("AUTOVID_FFMPEG_HWACCEL", "none")
    assert hwaccel_args() == []
    monkeypatch.setenv("AUTOVID_FFMPEG_HWACCEL", "cuda")
    assert hwaccel_args() == ["-hwaccel", "cuda"]
    monkeypatch.delenv("AUTOVID_FFMPEG_HWACCEL")
    assert hwaccel_args()[0] == "-hwaccel"  # GPU decode is on by default
