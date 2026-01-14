"""
Output formatting utilities for video transcription.

This module provides functions for assembling HTML documents,
formatting timestamps, and extracting titles from content.
"""

import base64
import re
from pathlib import Path
from typing import List, Tuple, Union

from .video_utils import format_timestamp

# CSS styles shared across all HTML output
HTML_STYLES = """
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem;
            line-height: 1.6;
            color: #333;
        }
        h1, h2, h3 { color: #1a1a1a; }
        img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin: 1rem 0;
            display: block;
        }
        pre {
            background: #f5f5f5;
            padding: 1rem;
            border-radius: 4px;
            overflow-x: auto;
        }
        code {
            font-family: 'Consolas', 'Monaco', monospace;
        }
        .visual-note {
            background: #e8f4f8;
            padding: 0.5rem 1rem;
            border-left: 3px solid #0077b6;
            margin: 1rem 0;
            font-style: italic;
        }
        table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
        th, td { border: 1px solid #ddd; padding: 0.5rem; text-align: left; }
        th { background: #f5f5f5; }
"""


def extract_title_from_html(content: str) -> str:
    """
    Extract title from the first <h1> tag in HTML content.

    Args:
        content: HTML content string.

    Returns:
        Extracted title text, or empty string if no h1 found.
    """
    title_match = re.search(r'<h1[^>]*>([^<]+)</h1>', content, re.IGNORECASE)
    if title_match:
        return title_match.group(1).strip()
    return ""


def clean_video_name_for_title(video_name: str) -> str:
    """
    Clean up a video filename to create a readable title.

    Removes common prefixes (hashes, IDs, course codes) and
    converts underscores to spaces.

    Args:
        video_name: Video filename without extension.

    Returns:
        Cleaned title string.
    """
    # Remove ID_hash prefix (e.g., "abc123_def456_")
    title = re.sub(r'^[a-zA-Z0-9]{20,}_[a-f0-9]{32}_', '', video_name)
    # Remove version/format suffix (e.g., "_V1_MP4_720")
    title = re.sub(r'_V\d+_MP4_\d+$', '', title)
    # Remove course codes (e.g., "T-GCP-A_M01_L01_001_")
    title = re.sub(r'^T-[A-Z0-9]+-[A-Z]_M\d+_L\d+_\d+_', '', title)
    # Convert underscores to spaces and title case
    title = title.replace('_', ' ').title()
    return title if title else "Video Transcript"


def assemble_html(
    content: str,
    keyframes: List[Tuple[float, Path]],
    video_name: str
) -> str:
    """
    Assemble final HTML by replacing image placeholders with base64-encoded images.

    This function takes HTML content with {{IMAGE_N}} placeholders and replaces
    them with base64 data URIs of the actual keyframe images.

    Args:
        content: HTML content with {{IMAGE_N}} placeholders.
        keyframes: List of (timestamp, frame_path) tuples.
        video_name: Video filename for fallback title.

    Returns:
        Complete HTML document string.
    """
    # Create image data URIs
    for i, (ts, frame_path) in enumerate(keyframes):
        placeholder = f"{{{{IMAGE_{i+1}}}}}"

        if frame_path.exists():
            with open(frame_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            data_uri = f"data:image/png;base64,{image_data}"
            content = content.replace(placeholder, data_uri)

    # Extract title from first <h1> in content, or use fallback
    title = extract_title_from_html(content)
    if not title:
        title = clean_video_name_for_title(video_name)

    # Wrap in full HTML document
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>{HTML_STYLES}
    </style>
</head>
<body>
    {content}
</body>
</html>"""

    return html


def assemble_html_with_descriptions(
    content: str,
    keyframe_descriptions: List[Tuple[float, Path, str]],
    video_name: str
) -> str:
    """
    Assemble HTML for local_only pipeline with keyframe descriptions.

    Similar to assemble_html but takes keyframe descriptions tuples
    with format (timestamp, frame_path, description).

    Args:
        content: HTML content with {{IMAGE_N}} placeholders.
        keyframe_descriptions: List of (timestamp, frame_path, description) tuples.
        video_name: Video filename for fallback title.

    Returns:
        Complete HTML document string.
    """
    # Replace image placeholders
    for i, (ts, frame_path, desc) in enumerate(keyframe_descriptions, 1):
        placeholder = f"{{{{IMAGE_{i}}}}}"
        if frame_path.exists():
            with open(frame_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            data_uri = f"data:image/png;base64,{image_data}"
            content = content.replace(placeholder, data_uri)

    # Extract title from first h1
    title = extract_title_from_html(content)
    if not title:
        title = clean_video_name_for_title(video_name)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>{HTML_STYLES}
    </style>
</head>
<body>
{content}
</body>
</html>"""

    return html


def format_transcript_segments(
    segments: List[Tuple[float, float, str]]
) -> str:
    """
    Format transcript segments with timestamps.

    Args:
        segments: List of (start_sec, end_sec, text) tuples.

    Returns:
        Formatted transcript string with timestamps.
    """
    lines = []
    for start, end, text in segments:
        ts = format_timestamp(start)
        lines.append(f"[{ts}] {text}")
    return "\n".join(lines)
