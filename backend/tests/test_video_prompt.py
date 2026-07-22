from pathlib import Path
from types import TracebackType
from typing import Any

import pytest

from prompts.create import video


def test_video_frame_times_default_can_cover_twenty_frames() -> None:
    times = video._frame_times(duration=30.0, count=20)

    assert len(times) == 20
    assert times[0] == 0.1
    assert times[-1] == 29.95


def test_video_frame_times_caps_at_provider_safe_limit() -> None:
    times = video._frame_times(duration=30.0, count=40)

    assert len(times) == video.MAX_VIDEO_PROMPT_FRAMES


def test_fallback_frames_spread_across_duration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured_command: list[str] = []

    class TempDir:
        def __enter__(self) -> Path:
            return tmp_path

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None,
        ) -> None:
            return None

    class Proc:
        returncode = 0
        stdout = b""
        stderr = b""

    def fake_run(command: list[str], **_: Any) -> Proc:
        nonlocal captured_command
        captured_command = command
        output_pattern = command[-1]
        (tmp_path / "frame-001.jpg").write_bytes(b"jpeg")
        assert output_pattern.endswith("frame-%03d.jpg")
        return Proc()

    monkeypatch.setattr(video.tempfile, "TemporaryDirectory", TempDir)
    monkeypatch.setattr(video.subprocess, "run", fake_run)
    monkeypatch.setattr(video, "_ffmpeg_exe", lambda: "ffmpeg")
    monkeypatch.setattr(
        video, "_image_bytes_to_data_url", lambda _: "data:image/jpeg;base64,x"
    )

    result = video._extract_fallback_frames(
        video_path="/tmp/input.mp4",
        frame_count=20,
        duration=30.0,
    )

    assert result == ["data:image/jpeg;base64,x"]
    vf_index = captured_command.index("-vf") + 1
    assert captured_command[vf_index] == "fps=0.666667"
