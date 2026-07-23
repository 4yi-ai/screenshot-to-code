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


def test_scene_change_probe_parses_showinfo_times(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_command: list[str] = []

    class Proc:
        returncode = 0
        stdout = b""
        stderr = (
            b"[Parsed_showinfo_1] n:0 pts_time:1.25\n"
            b"[Parsed_showinfo_1] n:1 pts_time:4\n"
        )

    def fake_run(command: list[str], **_: Any) -> Proc:
        nonlocal captured_command
        captured_command = command
        return Proc()

    monkeypatch.setenv("VIDEO_SCENE_CHANGE_THRESHOLD", "0.2")
    monkeypatch.setattr(video.subprocess, "run", fake_run)
    monkeypatch.setattr(video, "_ffmpeg_exe", lambda: "ffmpeg")

    times = video._probe_scene_change_times("/tmp/input.mp4")

    assert times == [1.25, 4.0]
    vf_index = captured_command.index("-vf") + 1
    assert captured_command[vf_index] == "select=gt(scene\\,0.200),showinfo"


def test_keyframe_probe_parses_showinfo_times(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_command: list[str] = []

    class Proc:
        returncode = 0
        stdout = b""
        stderr = b"[Parsed_showinfo_1] n:0 pts_time:0\n"

    def fake_run(command: list[str], **_: Any) -> Proc:
        nonlocal captured_command
        captured_command = command
        return Proc()

    monkeypatch.setattr(video.subprocess, "run", fake_run)
    monkeypatch.setattr(video, "_ffmpeg_exe", lambda: "ffmpeg")

    assert video._probe_keyframe_times("/tmp/input.mp4") == [0.0]
    vf_index = captured_command.index("-vf") + 1
    assert captured_command[vf_index] == "select=eq(pict_type\\,I),showinfo"


def test_merge_frame_times_prefers_scene_changes_within_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VIDEO_FRAME_MIN_GAP_SECONDS", "0.5")

    times = video._merge_frame_times(
        duration=10.0,
        count=20,
        scene_times=[1.0],
        keyframe_times=[1.1],
    )

    assert 1.0 in times
    assert 1.1 not in times


def test_merge_frame_times_caps_at_provider_limit() -> None:
    times = video._merge_frame_times(
        duration=30.0,
        count=40,
        scene_times=[float(index) for index in range(30)],
        keyframe_times=[],
    )

    assert len(times) == video.MAX_VIDEO_PROMPT_FRAMES
    assert times[0] == 0.0
    assert times[-1] == 29.95


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
