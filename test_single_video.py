#!/usr/bin/env python3
"""
Test script to diagnose video decoding issues.
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Check which video decoder is available
print("=== Checking video decoders ===")

try:
    import torchcodec
    print("✓ torchcodec is installed")
except ImportError:
    print("✗ torchcodec is NOT installed")

try:
    import decord
    print(f"✓ decord is installed (version: {decord.__version__})")
except ImportError:
    print("✗ decord is NOT installed")

try:
    import av
    print(f"✓ av (PyAV) is installed (version: {av.__version__})")
except ImportError:
    print("✗ av (PyAV) is NOT installed")

# Check transformers video loading
print("\n=== Checking transformers video utils ===")
try:
    from qwen_vl_utils import process_vision_info
    print("✓ qwen_vl_utils available")
except ImportError as e:
    print(f"✗ qwen_vl_utils not available: {e}")

# Try to load a video
print("\n=== Testing video loading ===")

from s3_handler import S3Handler

s3_handler = S3Handler()
s3_url = "s3://longcat-venus-diffusion-training-data/videos/"

print("Listing videos from S3...")
videos = s3_handler.list_video_files(s3_url)
print(f"Found {len(videos)} videos")

# Get a video from the END (where crashes happen)
if len(videos) > 100:
    test_video = videos[-100]  # 100th from the end
    print(f"\nTesting video from END: {test_video['name']}")
    print(f"Key: {test_video['key']}")

    # Try to download and decode
    import tempfile
    import urllib.request

    print("\nDownloading video...")
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        temp_path = f.name

    try:
        urllib.request.urlretrieve(test_video['url'], temp_path)
        file_size = os.path.getsize(temp_path)
        print(f"Downloaded to {temp_path} ({file_size} bytes)")

        # Try decord
        print("\nTrying decord...")
        try:
            from decord import VideoReader, cpu
            vr = VideoReader(temp_path, ctx=cpu(0))
            print(f"✓ decord loaded video: {len(vr)} frames, {vr.get_avg_fps()} fps")
        except Exception as e:
            print(f"✗ decord failed: {e}")

        # Try av
        print("\nTrying av (PyAV)...")
        try:
            import av
            container = av.open(temp_path)
            video_stream = container.streams.video[0]
            print(f"✓ av loaded video: {video_stream.frames} frames, codec={video_stream.codec.name}, {video_stream.width}x{video_stream.height}")
            container.close()
        except Exception as e:
            print(f"✗ av failed: {e}")

        # Try ffprobe for info
        print("\nGetting video info with ffprobe...")
        import subprocess
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_format', '-show_streams', temp_path],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            # Extract key info
            for line in result.stdout.split('\n'):
                if any(x in line.lower() for x in ['codec', 'width', 'height', 'duration', 'bit_rate']):
                    print(f"  {line}")
        else:
            print(f"ffprobe error: {result.stderr}")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Also test a video from the BEGINNING (where main.py works)
if len(videos) > 100:
    test_video = videos[100]  # 100th from the beginning
    print(f"\n\n=== Testing video from BEGINNING (should work): {test_video['name']} ===")
    print(f"Key: {test_video['key']}")

    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        temp_path = f.name

    try:
        urllib.request.urlretrieve(test_video['url'], temp_path)
        file_size = os.path.getsize(temp_path)
        print(f"Downloaded ({file_size} bytes)")

        # Try decord
        try:
            from decord import VideoReader, cpu
            vr = VideoReader(temp_path, ctx=cpu(0))
            print(f"✓ decord: {len(vr)} frames, {vr.get_avg_fps()} fps")
        except Exception as e:
            print(f"✗ decord failed: {e}")

        # Try av
        try:
            import av
            container = av.open(temp_path)
            video_stream = container.streams.video[0]
            print(f"✓ av: {video_stream.frames} frames, codec={video_stream.codec.name}, {video_stream.width}x{video_stream.height}")
            container.close()
        except Exception as e:
            print(f"✗ av failed: {e}")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

print("\n=== Done ===")
