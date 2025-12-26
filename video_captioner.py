"""
Video captioner module using Qwen2.5-VL model.
"""

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
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
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

        # Load model and processor
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device,
            attn_implementation="flash_attention_2",
        )
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(model_name)

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
        if prompt is None:
            prompt = self.DEFAULT_PROMPT

        # Prepare conversation with video
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "url": video_url},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Process inputs using chat template
        inputs = self.processor.apply_chat_template(
            conversation,
            fps=fps,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Move inputs to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Generate caption
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # Decode output
        generated_ids = [
            output_ids[i][len(inputs["input_ids"][i]) :]
            for i in range(len(output_ids))
        ]
        caption = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]

        return caption.strip()

    def __del__(self):
        """Clean up model resources."""
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "processor"):
            del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
