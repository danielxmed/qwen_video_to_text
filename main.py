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
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process, Queue, get_context
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


def gpu_worker(
    gpu_id: int,
    task_queue: Queue,
    result_queue: Queue,
    prompt: Optional[str],
    fps: float,
    max_tokens: int,
    skip_existing: bool,
    batch_size: int = 1,
):
    """
    Persistent GPU worker with prefetching for optimal GPU utilization.

    Downloads next batch while current batch is being processed by GPU.
    """
    load_dotenv()

    device = f"cuda:{gpu_id}"
    s3_handler = S3Handler()

    # Load model ONCE for this GPU
    logger.info(f"[GPU {gpu_id}] Loading model (batch_size={batch_size}, prefetch=ON)...")
    try:
        captioner = VideoCaptioner(device=device)
        logger.info(f"[GPU {gpu_id}] Model loaded successfully")
    except Exception as e:
        logger.error(f"[GPU {gpu_id}] Failed to load model: {e}")
        while True:
            task = task_queue.get()
            if task is None:
                break
            result_queue.put({
                "video_key": task["key"],
                "video_name": task["name"],
                "success": False,
                "error": f"Model failed to load: {e}",
            })
        return

    def collect_batch():
        """Collect a batch of tasks from queue, filtering skipped ones."""
        batch_tasks = []
        got_poison = False

        for _ in range(batch_size):
            task = task_queue.get()
            if task is None:
                got_poison = True
                break
            batch_tasks.append(task)

        # Filter out already processed videos
        tasks_to_process = []
        for video_info in batch_tasks:
            video_key = video_info["key"]
            video_name = video_info["name"]

            if skip_existing and s3_handler.check_caption_exists(video_info["bucket"], video_key):
                logger.info(f"[GPU {gpu_id}] Skipping {video_name} - caption exists")
                result_queue.put({
                    "video_key": video_key,
                    "video_name": video_name,
                    "success": True,
                    "caption_key": None,
                    "error": None,
                    "skipped": True,
                })
            else:
                tasks_to_process.append(video_info)

        return tasks_to_process, got_poison

    def preprocess_task(tasks):
        """Preprocess a batch (download videos, extract frames)."""
        if not tasks:
            return None, tasks
        video_urls = [t["url"] for t in tasks]
        preprocessed = captioner.preprocess_batch(video_urls, prompt, fps)
        return preprocessed, tasks

    def process_batch(preprocessed, tasks):
        """Run inference and upload results."""
        if not tasks or preprocessed is None:
            return

        video_names = [t["name"] for t in tasks]
        logger.info(f"[GPU {gpu_id}] Running inference on batch of {len(tasks)}: {video_names}")

        try:
            # Run inference (GPU work)
            captions, timing = captioner.run_inference(preprocessed, max_tokens)

            # Log timing
            total_time = sum(timing.values())
            if total_time > 0:
                logger.info(
                    f"[GPU {gpu_id}] Timing: "
                    f"preprocess={timing.get('preprocess', 0):.2f}s ({timing.get('preprocess', 0)/total_time*100:.1f}%), "
                    f"transfer={timing.get('transfer', 0):.2f}s ({timing.get('transfer', 0)/total_time*100:.1f}%), "
                    f"inference={timing.get('inference', 0):.2f}s ({timing.get('inference', 0)/total_time*100:.1f}%), "
                    f"decode={timing.get('decode', 0):.2f}s ({timing.get('decode', 0)/total_time*100:.1f}%)"
                )

            # Log sample caption for quality monitoring
            if captions:
                sample_name = tasks[0]["name"]
                sample_caption = captions[0][:200] + "..." if len(captions[0]) > 200 else captions[0]
                logger.info(f"[GPU {gpu_id}] Sample [{sample_name}]: {sample_caption}")

            # Upload results
            t_start_upload = time.time()
            for video_info, caption in zip(tasks, captions):
                video_key = video_info["key"]
                video_name = video_info["name"]

                try:
                    caption_key = s3_handler.upload_caption(
                        bucket=video_info["bucket"],
                        video_key=video_key,
                        caption=caption,
                    )
                    result_queue.put({
                        "video_key": video_key,
                        "video_name": video_name,
                        "success": True,
                        "caption_key": caption_key,
                        "error": None,
                        "skipped": False,
                    })
                except Exception as e:
                    logger.error(f"[GPU {gpu_id}] Upload error {video_name}: {e}")
                    result_queue.put({
                        "video_key": video_key,
                        "video_name": video_name,
                        "success": False,
                        "caption_key": None,
                        "error": str(e),
                        "skipped": False,
                    })

            logger.info(f"[GPU {gpu_id}] Upload: {time.time() - t_start_upload:.2f}s for {len(tasks)} captions")

        except Exception as e:
            logger.error(f"[GPU {gpu_id}] Batch error: {e}")
            for video_info in tasks:
                result_queue.put({
                    "video_key": video_info["key"],
                    "video_name": video_info["name"],
                    "success": False,
                    "caption_key": None,
                    "error": str(e),
                    "skipped": False,
                })

    # Main loop with prefetching
    with ThreadPoolExecutor(max_workers=1) as executor:
        # Get first batch with actual work (skip empty batches)
        current_tasks = []
        got_poison = False
        skipped_count = 0
        while not current_tasks and not got_poison:
            current_tasks, got_poison = collect_batch()
            if not current_tasks and not got_poison:
                skipped_count += batch_size
                if skipped_count % 1000 == 0:
                    logger.info(f"[GPU {gpu_id}] Skipped {skipped_count} videos so far, searching for work...")

        if not current_tasks:
            logger.info(f"[GPU {gpu_id}] No tasks to process (skipped {skipped_count}), exiting")
            return

        # Preprocess first batch synchronously
        logger.info(f"[GPU {gpu_id}] Found first batch of {len(current_tasks)} after skipping {skipped_count}")
        current_preprocessed, current_tasks = preprocess_task(current_tasks)
        prefetch_future = None

        while True:
            # Start prefetch for next batch in background (if we have a future from previous iteration)
            # The future was submitted at end of previous iteration

            # Process current batch (GPU inference) - runs in parallel with prefetch
            process_batch(current_preprocessed, current_tasks)

            # Now collect and prepare next batch
            if prefetch_future:
                # Wait for prefetch that was running during GPU work
                current_preprocessed, current_tasks = prefetch_future.result()
                prefetch_future = None
            else:
                current_preprocessed, current_tasks = None, []

            # Check if we're done
            if got_poison and not current_tasks:
                break

            # If no current tasks, need to collect more
            if not current_tasks:
                while not current_tasks and not got_poison:
                    current_tasks, got_poison = collect_batch()
                    if not current_tasks and not got_poison:
                        skipped_count += batch_size

                if not current_tasks:
                    break

                # Preprocess synchronously since we don't have a prefetch ready
                current_preprocessed, current_tasks = preprocess_task(current_tasks)

            # Submit prefetch for the batch AFTER next (to run during next GPU work)
            if not got_poison:
                next_tasks = []
                # Try to get next batch (non-blocking collection)
                next_tasks, got_poison = collect_batch()
                # Skip empty batches
                while not next_tasks and not got_poison:
                    next_tasks, got_poison = collect_batch()
                    if not next_tasks and not got_poison:
                        skipped_count += batch_size

                if next_tasks:
                    logger.info(f"[GPU {gpu_id}] Prefetching batch of {len(next_tasks)}...")
                    prefetch_future = executor.submit(preprocess_task, next_tasks)

    # Cleanup
    del captioner
    torch.cuda.empty_cache()
    logger.info(f"[GPU {gpu_id}] Worker finished")


def process_videos_parallel(
    s3_url: str,
    prompt: Optional[str] = None,
    fps: float = 1.0,
    max_tokens: int = 512,
    num_gpus: Optional[int] = None,
    skip_existing: bool = True,
    batch_size: int = 1,
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
        batch_size: Number of videos to process per batch per GPU

    Returns:
        List of processing results
    """
    # Get available GPUs
    available_gpus = get_available_gpus()
    if not available_gpus:
        raise RuntimeError("No GPUs available. This system requires CUDA-capable GPUs.")

    if num_gpus is not None:
        available_gpus = available_gpus[:num_gpus]

    logger.info(f"Using {len(available_gpus)} GPU(s): {available_gpus}, batch_size={batch_size}")

    # List videos from S3
    s3_handler = S3Handler()
    videos = s3_handler.list_video_files(s3_url)

    if not videos:
        logger.warning(f"No .mp4 videos found in {s3_url}")
        return []

    logger.info(f"Found {len(videos)} video(s) to process")

    # Use spawn context for CUDA compatibility
    ctx = get_context("spawn")

    # Create queues
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()

    # Start worker processes (one per GPU)
    workers = []
    for gpu_id in available_gpus:
        p = ctx.Process(
            target=gpu_worker,
            args=(gpu_id, task_queue, result_queue, prompt, fps, max_tokens, skip_existing, batch_size),
        )
        p.start()
        workers.append(p)

    # Submit all tasks to the queue
    for video in videos:
        task_queue.put(video)

    # Send poison pills to stop workers (one per worker)
    for _ in workers:
        task_queue.put(None)

    # Collect results
    results = []
    for _ in range(len(videos)):
        result = result_queue.get()
        results.append(result)

    # Wait for all workers to finish
    for p in workers:
        p.join()

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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Number of videos to process per batch per GPU (default: 2)",
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
            batch_size=args.batch_size,
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
