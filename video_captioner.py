"""
Video captioner module using Qwen2.5-VL model.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


class VideoCaptioner:
    """Generates captions for videos using Qwen2.5-VL model."""

    DEFAULT_PROMPT = (
        "Please provide a detailed description of this video. "
        "Describe what is happening, the main subjects, actions, "
        "setting, and any notable visual elements."
    )

    def __init__(
        self,
        model_name: str = "huihui-ai/Qwen2.5-VL-7B-Instruct-abliterated",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize the video captioner.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run the model on (cuda, cuda:0, cuda:1, etc.)
            dtype: Data type for model weights
        """
        self.device = device
        self.dtype = dtype
        self.model_name = model_name

        # Get HuggingFace token from environment
        hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")

        # Load model and processor
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device,
            attn_implementation="flash_attention_2",
            token=hf_token,
        )
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(model_name, token=hf_token)

    def generate_caption(
        self,
        video_url: str,
        prompt: str = None,
        max_new_tokens: int = 512,
        fps: float = 1.0,
    ) -> str:
        """
        Generate a caption for a video from URL.

        Args:
            video_url: URL to the video file (presigned S3 URL)
            prompt: Custom prompt for caption generation
            max_new_tokens: Maximum tokens to generate
            fps: Frames per second to sample from video

        Returns:
            Generated caption text
        """
        results = self.generate_captions_batch(
            video_urls=[video_url],
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            fps=fps,
        )
        return results[0]

    def preprocess_batch(
        self,
        video_urls: list,
        prompt: str = None,
        fps: float = 1.0,
    ) -> dict:
        """
        Preprocess videos: download and extract frames (CPU/IO bound).
        Can be run in a background thread while GPU is busy.

        Args:
            video_urls: List of URLs to video files
            prompt: Custom prompt for caption generation
            fps: Frames per second to sample from video

        Returns:
            Dict with preprocessed data ready for run_inference
        """
        if prompt is None:
            prompt = self.DEFAULT_PROMPT

        if not video_urls:
            return {"empty": True, "num_videos": 0}

        t_start = time.time()

        # Prepare conversations for all videos
        conversations = []
        for video_url in video_urls:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "url": video_url},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            conversations.append(conversation)

        # Process all inputs in parallel (downloads videos and extracts frames)
        def process_single(idx_conv):
            idx, conversation = idx_conv
            inputs = self.processor.apply_chat_template(
                conversation,
                fps=fps,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            return idx, inputs

        batch_inputs = [None] * len(conversations)
        with ThreadPoolExecutor(max_workers=len(conversations)) as executor:
            futures = [executor.submit(process_single, (i, conv)) for i, conv in enumerate(conversations)]
            for future in as_completed(futures):
                idx, inputs = future.result()
                batch_inputs[idx] = inputs

        # Pad inputs to same length for batching
        max_len = max(inp["input_ids"].shape[1] for inp in batch_inputs)

        padded_input_ids = []
        padded_attention_mask = []
        pixel_values_list = []
        video_grid_thw_list = []

        for inputs in batch_inputs:
            seq_len = inputs["input_ids"].shape[1]
            padding_len = max_len - seq_len

            # Pad input_ids (left padding for generation)
            if padding_len > 0:
                pad_ids = torch.full((1, padding_len), self.processor.tokenizer.pad_token_id, dtype=inputs["input_ids"].dtype)
                padded_input_ids.append(torch.cat([pad_ids, inputs["input_ids"]], dim=1))
                pad_mask = torch.zeros((1, padding_len), dtype=inputs["attention_mask"].dtype)
                padded_attention_mask.append(torch.cat([pad_mask, inputs["attention_mask"]], dim=1))
            else:
                padded_input_ids.append(inputs["input_ids"])
                padded_attention_mask.append(inputs["attention_mask"])

            # Collect pixel values and grid info
            if "pixel_values_videos" in inputs:
                pixel_values_list.append(inputs["pixel_values_videos"])
                video_grid_thw_list.append(inputs["video_grid_thw"])

        preprocess_time = time.time() - t_start

        return {
            "empty": False,
            "num_videos": len(video_urls),
            "max_len": max_len,
            "padded_input_ids": padded_input_ids,
            "padded_attention_mask": padded_attention_mask,
            "pixel_values_list": pixel_values_list,
            "video_grid_thw_list": video_grid_thw_list,
            "preprocess_time": preprocess_time,
        }

    def run_inference(
        self,
        preprocessed: dict,
        max_new_tokens: int = 512,
    ) -> tuple:
        """
        Run inference on preprocessed data (GPU bound).

        Args:
            preprocessed: Output from preprocess_batch
            max_new_tokens: Maximum tokens to generate

        Returns:
            Tuple of (captions list, timing dict)
        """
        timing = {"preprocess": preprocessed.get("preprocess_time", 0)}

        if preprocessed.get("empty", True):
            return [], timing

        num_videos = preprocessed["num_videos"]
        max_len = preprocessed["max_len"]

        # Transfer to GPU
        t_start_transfer = time.time()
        batched_inputs = {
            "input_ids": torch.cat(preprocessed["padded_input_ids"], dim=0).to(self.device),
            "attention_mask": torch.cat(preprocessed["padded_attention_mask"], dim=0).to(self.device),
        }

        if preprocessed["pixel_values_list"]:
            batched_inputs["pixel_values_videos"] = torch.cat(preprocessed["pixel_values_list"], dim=0).to(self.device)
            batched_inputs["video_grid_thw"] = torch.cat(preprocessed["video_grid_thw_list"], dim=0).to(self.device)
        torch.cuda.synchronize(self.device)
        timing["transfer"] = time.time() - t_start_transfer

        # Generate captions
        t_start_inference = time.time()
        with torch.no_grad():
            output_ids = self.model.generate(
                **batched_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )
        torch.cuda.synchronize(self.device)
        timing["inference"] = time.time() - t_start_inference

        # Decode outputs
        t_start_decode = time.time()
        captions = []
        for i in range(num_videos):
            generated_ids = output_ids[i][max_len:]
            caption = self.processor.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            captions.append(caption.strip())
        timing["decode"] = time.time() - t_start_decode

        return captions, timing

    def generate_captions_batch(
        self,
        video_urls: list,
        prompt: str = None,
        max_new_tokens: int = 512,
        fps: float = 1.0,
        return_timing: bool = False,
    ) -> tuple:
        """
        Generate captions for multiple videos in a batch.
        Convenience method that combines preprocess_batch and run_inference.

        Args:
            video_urls: List of URLs to video files (presigned S3 URLs)
            prompt: Custom prompt for caption generation
            max_new_tokens: Maximum tokens to generate
            fps: Frames per second to sample from video
            return_timing: If True, return timing information

        Returns:
            List of generated caption texts, or (captions, timing_dict) if return_timing=True
        """
        preprocessed = self.preprocess_batch(video_urls, prompt, fps)
        captions, timing = self.run_inference(preprocessed, max_new_tokens)

        if return_timing:
            return captions, timing
        return captions

    def __del__(self):
        """Clean up model resources."""
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "processor"):
            del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
