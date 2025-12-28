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
    init_delay: float = 0.0,
):
    """
    Persistent GPU worker with prefetching for optimal GPU utilization.

    Downloads next batch while current batch is being processed by GPU.
    """
    import traceback
    import sys

    # Staggered initialization to avoid race conditions
    if init_delay > 0:
        time.sleep(init_delay)

    load_dotenv()

    device = f"cuda:{gpu_id}"
    s3_handler = S3Handler()

    # Load model ONCE for this GPU
    logger.info(f"[GPU {gpu_id}] Loading model (batch_size={batch_size}, prefetch=ON)...")
    try:
        captioner = VideoCaptioner(device=device)
        logger.info(f"[GPU {gpu_id}] Model loaded successfully")
        # Signal successful initialization
        result_queue.put({"type": "init", "gpu_id": gpu_id, "success": True})
    except Exception as e:
        error_tb = traceback.format_exc()
        logger.error(f"[GPU {gpu_id}] Failed to load model: {e}\n{error_tb}")
        # Signal failed initialization
        result_queue.put({"type": "init", "gpu_id": gpu_id, "success": False, "error": str(e)})
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

        video_names = [t["name"] for t in tasks]
        logger.info(f"[GPU {gpu_id}] Preprocessing batch: {video_names}")

        video_urls = [t["url"] for t in tasks]
        preprocessed = captioner.preprocess_batch(video_urls, prompt, fps)

        # Handle failed videos from preprocessing
        failed_indices = preprocessed.get("failed_indices", [])
        if failed_indices:
            for idx, error in failed_indices:
                video_info = tasks[idx]
                logger.warning(f"[GPU {gpu_id}] Preprocessing failed for {video_info['name']}: {error}")
                result_queue.put({
                    "video_key": video_info["key"],
                    "video_name": video_info["name"],
                    "success": False,
                    "caption_key": None,
                    "error": error,
                    "skipped": False,
                })

            # Filter tasks to only include successfully preprocessed videos
            valid_indices = preprocessed.get("valid_indices", [])
            tasks = [tasks[i] for i in valid_indices]
            logger.info(f"[GPU {gpu_id}] {len(failed_indices)} videos failed, {len(tasks)} remaining in batch")

        if preprocessed.get("empty", False):
            logger.info(f"[GPU {gpu_id}] All videos in batch failed preprocessing")
            return None, []

        logger.info(f"[GPU {gpu_id}] Preprocessing complete for batch: {[t['name'] for t in tasks]}")

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

    # Sequential initialization timeout per GPU
    INIT_TIMEOUT_PER_GPU = 120  # 2 minutes per GPU
    MAX_RETRIES = 3

    def start_worker(gpu_id: int, delay: float = 0.0) -> Process:
        """Start a worker process for a GPU."""
        p = ctx.Process(
            target=gpu_worker,
            args=(gpu_id, task_queue, result_queue, prompt, fps, max_tokens, skip_existing, batch_size, delay),
        )
        p.start()
        return p

    def wait_for_worker_init(gpu_id: int, process: Process, timeout: float) -> bool:
        """Wait for a worker to initialize. Returns True if successful."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            # Check if process died
            if not process.is_alive():
                logger.error(f"[GPU {gpu_id}] Worker died during initialization (exit code: {process.exitcode})")
                return False

            # Check for init message
            try:
                msg = result_queue.get(timeout=2.0)
                if msg.get("type") == "init" and msg["gpu_id"] == gpu_id:
                    if msg["success"]:
                        return True
                    else:
                        logger.error(f"[GPU {gpu_id}] Initialization failed: {msg.get('error')}")
                        return False
                else:
                    # Put back messages from other GPUs
                    result_queue.put(msg)
            except:
                pass  # Queue timeout, continue

        logger.error(f"[GPU {gpu_id}] Initialization timeout after {timeout}s")
        return False

    # Sequential initialization: start each GPU one at a time
    workers = {}  # gpu_id -> Process
    initialized_gpus = set()
    failed_gpus = set()

    logger.info(f"Starting {len(available_gpus)} workers with SEQUENTIAL initialization...")

    for gpu_id in available_gpus:
        success = False

        for retry in range(MAX_RETRIES):
            if retry > 0:
                logger.info(f"[GPU {gpu_id}] Retry {retry}/{MAX_RETRIES}...")
                time.sleep(5)  # Brief pause before retry

            logger.info(f"[GPU {gpu_id}] Starting worker (attempt {retry + 1}/{MAX_RETRIES})...")

            # Kill old process if exists
            if gpu_id in workers:
                old_p = workers[gpu_id]
                if old_p.is_alive():
                    old_p.terminate()
                    old_p.join(timeout=5)

            # Start worker
            workers[gpu_id] = start_worker(gpu_id, delay=2.0)  # Small delay for stability

            # Wait for initialization
            if wait_for_worker_init(gpu_id, workers[gpu_id], INIT_TIMEOUT_PER_GPU):
                logger.info(f"[GPU {gpu_id}] Initialized successfully ({len(initialized_gpus) + 1}/{len(available_gpus)})")
                initialized_gpus.add(gpu_id)
                success = True
                break

        if not success:
            logger.error(f"[GPU {gpu_id}] Failed after {MAX_RETRIES} attempts, skipping")
            failed_gpus.add(gpu_id)
            # Kill the failed process
            if gpu_id in workers and workers[gpu_id].is_alive():
                workers[gpu_id].terminate()
                workers[gpu_id].join(timeout=5)

    if failed_gpus:
        logger.warning(f"GPUs {failed_gpus} failed to initialize and will be skipped")

    active_workers = {gpu_id: p for gpu_id, p in workers.items() if gpu_id in initialized_gpus}
    if not active_workers:
        raise RuntimeError("All GPU workers failed to initialize!")

    logger.info(f"Successfully initialized {len(active_workers)} workers: GPUs {list(active_workers.keys())}")

    # Submit all tasks to the queue
    for video in videos:
        task_queue.put(video)

    # Send poison pills to stop workers (one per active worker)
    for _ in active_workers:
        task_queue.put(None)

    # Collect results with worker health monitoring and auto-restart
    results = []
    expected_results = len(videos)
    check_interval = 10  # Check worker health every 10 seconds
    last_check = time.time()
    worker_restart_count = {}  # gpu_id -> restart count
    MAX_RUNTIME_RESTARTS = 3

    while len(results) < expected_results:
        # Check worker health periodically
        if time.time() - last_check > check_interval:
            last_check = time.time()

            for gpu_id in list(active_workers.keys()):
                p = active_workers[gpu_id]
                if not p.is_alive():
                    restart_count = worker_restart_count.get(gpu_id, 0)

                    if restart_count < MAX_RUNTIME_RESTARTS:
                        logger.warning(f"[GPU {gpu_id}] Worker died (exit code: {p.exitcode}), restarting ({restart_count + 1}/{MAX_RUNTIME_RESTARTS})...")

                        # Start new worker
                        new_worker = start_worker(gpu_id, delay=5.0)
                        workers[gpu_id] = new_worker
                        active_workers[gpu_id] = new_worker
                        worker_restart_count[gpu_id] = restart_count + 1

                        # Wait for init (with shorter timeout for runtime restart)
                        if wait_for_worker_init(gpu_id, new_worker, timeout=90):
                            logger.info(f"[GPU {gpu_id}] Runtime restart successful!")
                            # Add new poison pill for this worker
                            task_queue.put(None)
                        else:
                            logger.error(f"[GPU {gpu_id}] Runtime restart failed")
                            del active_workers[gpu_id]
                    else:
                        logger.error(f"[GPU {gpu_id}] Worker died too many times, giving up")
                        if gpu_id in active_workers:
                            del active_workers[gpu_id]

            if not active_workers:
                logger.error("All workers have died! Aborting...")
                break

        try:
            result = result_queue.get(timeout=5.0)
            # Skip init messages that might be late
            if result.get("type") == "init":
                continue
            results.append(result)
            if len(results) % 100 == 0:
                logger.info(f"Progress: {len(results)}/{expected_results} results collected")
        except:
            pass  # Queue timeout, continue

    # Wait for all workers to finish
    for p in workers.values():
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()

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
