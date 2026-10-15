#!/usr/bin/env python3
"""Download one video with yt-dlp, keeping the original video quality.

Install the downloader:
    python -m pip install -U "yt-dlp[default]"

Example:
    python download_video.py "https://www.youtube.com/watch?v=jH7XWFT41L4"
    python download_video.py URL -o downloads --cookies-from-browser firefox

ffmpeg is needed to merge separate video and audio streams at full quality.
Without ffmpeg, the script requests a file that already contains both streams.
"""

import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("The height must be greater than zero.")
    return number


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("url", help="Video URL")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("downloads"),
                        help="Destination directory (default: downloads)")
    parser.add_argument("--max-height", type=positive_int,
                        help="Limit resolution, for example 1080; default: best available")
    cookies = parser.add_mutually_exclusive_group()
    cookies.add_argument("--cookies-from-browser",
                         help="Browser used for YouTube login, for example firefox or chrome")
    cookies.add_argument("--cookies", type=Path,
                         help="Path to an exported Netscape-format cookies file")
    args = parser.parse_args()

    parsed_url = urlparse(args.url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.hostname:
        parser.error("Provide a complete http:// or https:// video URL.")
    if args.cookies is not None:
        args.cookies = args.cookies.expanduser().resolve()
        if not args.cookies.is_file():
            parser.error("The cookies file does not exist: {}".format(args.cookies))

    if importlib.util.find_spec("yt_dlp") is not None:
        command = [sys.executable, "-m", "yt_dlp"]
    elif shutil.which("yt-dlp"):
        command = [shutil.which("yt-dlp")]
    else:
        print('Install the downloader: python -m pip install -U "yt-dlp[default]"',
              file=sys.stderr)
        return 1

    destination = args.output_dir.expanduser().resolve()
    try:
        destination.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        print("Cannot create the destination directory: {}".format(error), file=sys.stderr)
        return 1

    has_ffmpeg = shutil.which("ffmpeg") is not None
    height_filter = "[height<={}]".format(args.max_height) if args.max_height else ""
    if has_ffmpeg:
        video_format = "bv*{0}+ba/b{0}".format(height_filter)
    else:
        video_format = "b{}".format(height_filter)
        print("ffmpeg was not found; available video quality may be lower.", file=sys.stderr)

    command += [
        "--ignore-config",
        "--no-playlist",
        "--no-overwrites",
        "--continue",
        "--windows-filenames",
        "--socket-timeout", "20",
        "--retries", "3",
        "--fragment-retries", "3",
        "--newline",
        "--progress",
        "--format", video_format,
        "--paths", str(destination),
        "--output", "%(title).150s [%(id)s].%(ext)s",
        "--print", "after_move:filepath",
        "--no-simulate",
    ]
    if has_ffmpeg:
        command += ["--merge-output-format", "mp4/mkv"]
    if args.cookies_from_browser:
        command += ["--cookies-from-browser", args.cookies_from_browser]
    elif args.cookies:
        command += ["--cookies", str(args.cookies)]
    command += ["--", args.url]

    print("Downloading to: {}".format(destination), flush=True)
    try:
        result = subprocess.run(command, check=False)
    except KeyboardInterrupt:
        print("\nDownload interrupted. Run the same command to resume.", file=sys.stderr)
        return 130
    except OSError as error:
        print("Cannot start yt-dlp: {}".format(error), file=sys.stderr)
        return 1

    if result.returncode:
        print('Download failed. Try updating: python -m pip install -U "yt-dlp[default]"',
              file=sys.stderr)
        print("If YouTube asks you to sign in, add --cookies-from-browser firefox "
              "or --cookies-from-browser chrome.", file=sys.stderr)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
