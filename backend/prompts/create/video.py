import base64
import io
import os
import re
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg
from openai.types.chat import ChatCompletionContentPartParam, ChatCompletionMessageParam
from PIL import Image

from prompts.prompt_types import Stack
from prompts import system_prompt
from prompts.design_system import build_design_system_prompt_block
from prompts.policies import build_selected_stack_policy, build_user_image_policy


def _video_suffix(video_data_url: str) -> str:
    mime_type = video_data_url.split(";", 1)[0].split(":", 1)[1].lower()
    if mime_type == "video/webm":
        return ".webm"
    if mime_type == "video/quicktime":
        return ".mov"
    return ".mp4"


def _decode_video_data_url(video_data_url: str) -> bytes:
    if "," not in video_data_url:
        raise ValueError("Video data URL is missing base64 payload")
    return base64.b64decode(video_data_url.split(",", 1)[1])


def _ffmpeg_exe() -> str:
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return os.environ.get("FFMPEG_BINARY", "ffmpeg")


def _probe_duration(video_path: str) -> float:
    proc = subprocess.run(
        [_ffmpeg_exe(), "-hide_banner", "-i", video_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    probe_output = (proc.stdout + proc.stderr).decode("utf-8", "replace")
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", probe_output)
    if not match:
        return 0.0
    hours, minutes, seconds = match.groups()
    return (int(hours) * 3600) + (int(minutes) * 60) + float(seconds)


def _frame_times(duration: float, count: int) -> list[float]:
    if not duration or duration <= 0:
        return [0.1]

    frame_count = max(1, min(count, 8))
    if frame_count == 1:
        return [min(max(duration * 0.5, 0.05), max(duration - 0.05, 0))]

    start = min(0.1, max(duration * 0.1, 0))
    end = max(duration - 0.05, start)
    return [
        start + ((end - start) * index) / (frame_count - 1)
        for index in range(frame_count)
    ]


def _image_bytes_to_data_url(image_bytes: bytes) -> str:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    max_side = int(os.environ.get("VIDEO_FRAME_MAX_SIDE", "1024") or "1024")
    if max_side > 0:
        image.thumbnail((max_side, max_side))

    output = io.BytesIO()
    image.save(output, format="JPEG", quality=85, optimize=True)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _extract_frame_jpeg(video_path: str, time_seconds: float) -> bytes:
    ffmpeg = _ffmpeg_exe()
    seek_time = f"{max(time_seconds, 0):.3f}"
    commands = [
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            seek_time,
            "-i",
            video_path,
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "pipe:1",
        ],
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            video_path,
            "-ss",
            seek_time,
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "pipe:1",
        ],
    ]
    last_error = ""
    for command in commands:
        proc = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
        if proc.returncode == 0 and proc.stdout:
            return proc.stdout
        last_error = proc.stderr.decode("utf-8", "replace")
    raise ValueError(last_error or f"ffmpeg produced no frame at {seek_time}s")


def _extract_fallback_frames(video_path: str, frame_count: int) -> list[str]:
    capped_frame_count = max(1, min(frame_count, 8))
    with tempfile.TemporaryDirectory() as frame_dir:
        output_pattern = str(Path(frame_dir) / "frame-%03d.jpg")
        proc = subprocess.run(
            [
                _ffmpeg_exe(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                video_path,
                "-vf",
                f"fps={capped_frame_count}",
                "-frames:v",
                str(capped_frame_count),
                output_pattern,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=60,
        )
        if proc.returncode != 0:
            raise ValueError(proc.stderr.decode("utf-8", "replace"))
        return [
            _image_bytes_to_data_url(frame_path.read_bytes())
            for frame_path in sorted(Path(frame_dir).glob("frame-*.jpg"))
        ]


def _video_to_frame_data_urls(video_data_url: str) -> list[str]:
    if video_data_url.startswith("data:image/"):
        return [video_data_url]
    if not video_data_url.startswith("data:video/"):
        return [video_data_url]

    frame_count = int(os.environ.get("VIDEO_PROMPT_FRAME_COUNT", "4") or "4")
    suffix = _video_suffix(video_data_url)
    frame_urls: list[str] = []
    with tempfile.NamedTemporaryFile(suffix=suffix) as video_file:
        video_file.write(_decode_video_data_url(video_data_url))
        video_file.flush()
        duration = _probe_duration(video_file.name)
        for time in _frame_times(duration, frame_count):
            try:
                frame = _extract_frame_jpeg(video_file.name, time)
                frame_urls.append(_image_bytes_to_data_url(frame))
            except Exception as exc:
                print(f"[video_prompt] skipping unreadable frame at {time:.3f}s: {exc}")
        if not frame_urls:
            frame_urls = _extract_fallback_frames(video_file.name, frame_count)

    if not frame_urls:
        raise ValueError("Unable to extract image frames from uploaded video")
    return frame_urls


def build_video_prompt_messages(
    video_data_url: str,
    stack: Stack,
    text_prompt: str,
    image_generation_enabled: bool,
    design_system: str | None = None,
) -> list[ChatCompletionMessageParam]:
    frame_data_urls = _video_to_frame_data_urls(video_data_url)
    image_policy = build_user_image_policy(image_generation_enabled)
    selected_stack = build_selected_stack_policy(stack)
    design_system_block = build_design_system_prompt_block(design_system)
    user_text = f"""
    You have been given a sequence of frames from a video of a user interacting with a web app. Re-create the same app exactly such that the same user interactions will produce the same results in the app you build.

    - Treat the images in order as frames sampled from the interaction video.
    - Infer the UI state changes and interactions from the frame sequence.
    - Make sure the app looks exactly like what you see in the frames.
    - Pay close attention to background color, text color, font size, font family,
    padding, margin, border, etc. Match the colors and sizes exactly.
    - {image_policy}
    - If some functionality requires a backend call, just mock the data instead.
    - MAKE THE APP FUNCTIONAL using JavaScript. Allow the user to interact with the app and get the same behavior as shown in the frames.
    - Use SVGs and interactive 3D elements if needed to match the functionality shown in the frames.

    Analyze these frames and generate the code.

    {selected_stack}
    {design_system_block}
    """
    if text_prompt.strip():
        user_text = user_text + "\n\nAdditional instructions: " + text_prompt

    user_content: list[ChatCompletionContentPartParam] = []
    for frame_data_url in frame_data_urls:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": frame_data_url, "detail": "high"},
            }
        )
    user_content.append(
        {
            "type": "text",
            "text": user_text,
        }
    )

    return [
        {
            "role": "system",
            "content": system_prompt.SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]
