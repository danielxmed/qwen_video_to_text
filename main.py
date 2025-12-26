#!/usr/bin/env python3
"""
Video to Text Captioning System using Qwen2.5-VL.

Processes videos from an S3 directory, generates captions using Qwen2.5-VL model,
and uploads the caption files back to S3.
"""

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from typing import List, Optional

import torch
from dotenv import load_dotenv

from s3_handler import S3Handler
from video_captioner import VideoCaptioner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(processName)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def get_available_gpus() -> List[int]:
    """Get list of available GPU indices."""
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def process_video_on_gpu(
    video_info: dict,
    gpu_id: int,
    prompt: Optional[str] = None,
    fps: float = 1.0,
    max_tokens: int = 512,
    skip_existing: bool = True,
) -> dict:
    """
    Process a single video on a specific GPU.

    This function is designed to run in a separate process.

    Args:
        video_info: Dict with video metadata (key, url, name, bucket)
        gpu_id: GPU device index to use
        prompt: Custom prompt for caption generation
        fps: Frames per second to sample
        max_tokens: Maximum tokens to generate
        skip_existing: Skip if caption already exists

    Returns:
        Dict with processing result
    """
    # Load environment variables in subprocess
    load_dotenv()

    device = f"cuda:{gpu_id}"
    video_key = video_info["key"]
    video_name = video_info["name"]

    result = {
        "video_key": video_key,
        "video_name": video_name,
        "success": False,
        "caption_key": None,
        "error": None,
        "skipped": False,
    }

    try:
        # Check if caption already exists
        if skip_existing:
            s3_handler = S3Handler()
            if s3_handler.check_caption_exists(video_info["bucket"], video_key):
                logger.info(f"[GPU {gpu_id}] Skipping {video_name} - caption exists")
                result["skipped"] = True
                result["success"] = True
                return result

        logger.info(f"[GPU {gpu_id}] Processing: {video_name}")

        # Initialize captioner for this GPU
        captioner = VideoCaptioner(device=device)

        # Generate caption
        caption = captioner.generate_caption(
            video_url=video_info["url"],
            prompt=prompt,
            max_new_tokens=max_tokens,
            fps=fps,
        )

        logger.info(f"[GPU {gpu_id}] Generated caption for {video_name} ({len(caption)} chars)")

        # Upload caption to S3
        s3_handler = S3Handler()
        caption_key = s3_handler.upload_caption(
            bucket=video_info["bucket"],
            video_key=video_key,
            caption=caption,
        )

        result["success"] = True
        result["caption_key"] = caption_key
        logger.info(f"[GPU {gpu_id}] Uploaded: {caption_key}")

        # Cleanup
        del captioner
        torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"[GPU {gpu_id}] Error processing {video_name}: {e}")
        result["error"] = str(e)

    return result


def process_videos_parallel(
    s3_url: str,
    prompt: Optional[str] = None,
    fps: float = 1.0,
    max_tokens: int = 512,
    num_gpus: Optional[int] = None,
    skip_existing: bool = True,
) -> List[dict]:
    """
    Process all videos in an S3 directory using multiple GPUs in parallel.

    Args:
        s3_url: S3 URL to the directory containing videos
        prompt: Custom prompt for caption generation
        fps: Frames per second to sample from videos
        max_tokens: Maximum tokens to generate per caption
        num_gpus: Number of GPUs to use (None = all available)
        skip_existing: Skip videos that already have captions

    Returns:
        List of processing results
    """
    # Get available GPUs
    available_gpus = get_available_gpus()
    if not available_gpus:
        raise RuntimeError("No GPUs available. This system requires CUDA-capable GPUs.")

    if num_gpus is not None:
        available_gpus = available_gpus[:num_gpus]

    logger.info(f"Using {len(available_gpus)} GPU(s): {available_gpus}")

    # List videos from S3
    s3_handler = S3Handler()
    videos = s3_handler.list_video_files(s3_url)

    if not videos:
        logger.warning(f"No .mp4 videos found in {s3_url}")
        return []

    logger.info(f"Found {len(videos)} video(s) to process")

    results = []

    # Use spawn context for CUDA compatibility
    ctx = get_context("spawn")

    # Process videos in parallel across GPUs
    with ProcessPoolExecutor(
        max_workers=len(available_gpus),
        mp_context=ctx,
    ) as executor:
        # Submit jobs round-robin to GPUs
        futures = {}
        for i, video in enumerate(videos):
            gpu_id = available_gpus[i % len(available_gpus)]
            future = executor.submit(
                process_video_on_gpu,
                video,
                gpu_id,
                prompt,
                fps,
                max_tokens,
                skip_existing,
            )
            futures[future] = video

        # Collect results
        for future in as_completed(futures):
            video = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {video['name']}: {e}")
                results.append({
                    "video_key": video["key"],
                    "video_name": video["name"],
                    "success": False,
                    "error": str(e),
                })

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate captions for videos in S3 using Qwen2.5-VL"
    )
    parser.add_argument(
        "s3_url",
        type=str,
        help="S3 URL to directory containing videos (s3://bucket/path/ or https://...)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt for caption generation",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frames per second to sample from videos (default: 1.0)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate per caption (default: 512)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: all available)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Process videos even if caption already exists",
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Validate AWS credentials
    if not os.environ.get("AWS_ACCESS_KEY_ID") or not os.environ.get("AWS_SECRET_ACCESS_KEY"):
        logger.error("AWS credentials not found. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        sys.exit(1)

    logger.info(f"Starting video captioning for: {args.s3_url}")

    try:
        results = process_videos_parallel(
            s3_url=args.s3_url,
            prompt=args.prompt,
            fps=args.fps,
            max_tokens=args.max_tokens,
            num_gpus=args.num_gpus,
            skip_existing=not args.no_skip_existing,
        )

        # Summary
        successful = sum(1 for r in results if r["success"] and not r.get("skipped"))
        skipped = sum(1 for r in results if r.get("skipped"))
        failed = sum(1 for r in results if not r["success"])

        logger.info(f"\n{'='*50}")
        logger.info(f"Processing complete!")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Skipped (existing): {skipped}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"{'='*50}")

        if failed > 0:
            logger.info("\nFailed videos:")
            for r in results:
                if not r["success"]:
                    logger.info(f"  - {r['video_name']}: {r.get('error', 'Unknown error')}")

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
