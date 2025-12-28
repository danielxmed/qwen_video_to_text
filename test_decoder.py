#!/usr/bin/env python3
"""Test video decoders directly."""
import os
import sys
import tempfile
import urllib.request
from dotenv import load_dotenv

load_dotenv()

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

# Get a specific video URL directly
import boto3
from botocore.config import Config

config = Config(
    retries={"max_attempts": 3, "mode": "adaptive"},
)

s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    region_name=os.environ.get("AWS_REGION", "us-east-1"),
    config=config,
)

bucket = "longcat-venus-diffusion-training-data"

# Get ONE specific video that's crashing
test_keys = [
    "videos/ph63d025dfb8fe0_chunk_0300.mp4",  # Crashing video
    "videos/0000a38e8a4e40b7.mp4",  # From the beginning (should work)
]

for key in test_keys:
    print(f"\n{'='*60}")
    print(f"Testing: {key}")
    print('='*60)

    try:
        # Check if it exists and get size
        response = s3_client.head_object(Bucket=bucket, Key=key)
        size = response['ContentLength']
        print(f"Size: {size} bytes ({size/1024/1024:.2f} MB)")

        # Generate presigned URL
        url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=3600,
        )

        # Download
        print("Downloading...")
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            temp_path = f.name

        urllib.request.urlretrieve(url, temp_path)
        print(f"Downloaded to: {temp_path}")

        # Get info with ffprobe
        print("\nffprobe info:")
        import subprocess
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries',
             'stream=codec_name,width,height,duration,nb_frames',
             '-of', 'csv=p=0', temp_path],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"  {result.stdout.strip()}")
        else:
            print(f"  Error: {result.stderr}")

        # Try decord
        print("\nTrying decord...")
        try:
            from decord import VideoReader, cpu
            vr = VideoReader(temp_path, ctx=cpu(0))
            print(f"  ✓ Loaded: {len(vr)} frames, {vr.get_avg_fps():.2f} fps")
            # Try reading a frame
            frame = vr[0]
            print(f"  ✓ Frame 0: shape={frame.shape}")
            del vr
        except Exception as e:
            print(f"  ✗ Failed: {e}")

        # Try av
        print("\nTrying av (PyAV)...")
        try:
            import av
            container = av.open(temp_path)
            vs = container.streams.video[0]
            print(f"  ✓ Loaded: codec={vs.codec.name}, {vs.width}x{vs.height}")
            # Try reading a frame
            for frame in container.decode(video=0):
                print(f"  ✓ Frame: {frame.width}x{frame.height}")
                break
            container.close()
        except Exception as e:
            print(f"  ✗ Failed: {e}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'temp_path' in dir() and os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"\nCleaned up: {temp_path}")

print("\n=== Done ===")
