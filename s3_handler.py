"""
S3 Handler module for managing video files and captions in AWS S3.
"""

import os
import re
from urllib.parse import urlparse
from typing import List, Tuple

import boto3
from botocore.config import Config


class S3Handler:
    """Handles S3 operations for listing videos and uploading captions."""

    def __init__(self):
        """Initialize S3 client with credentials from environment."""
        config = Config(
            retries={"max_attempts": 3, "mode": "adaptive"},
            connect_timeout=30,
            read_timeout=60,
        )

        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
            endpoint_url=os.environ.get("AWS_ENDPOINT_URL"),
            config=config,
        )

    def parse_s3_url(self, s3_url: str) -> Tuple[str, str]:
        """
        Parse S3 URL to extract bucket name and prefix.

        Args:
            s3_url: S3 URL in format s3://bucket-name/path/to/directory/
                    or https://bucket-name.s3.region.amazonaws.com/path/

        Returns:
            Tuple of (bucket_name, prefix)
        """
        if s3_url.startswith("s3://"):
            parsed = urlparse(s3_url)
            bucket = parsed.netloc
            prefix = parsed.path.lstrip("/")
        elif "s3." in s3_url and "amazonaws.com" in s3_url:
            parsed = urlparse(s3_url)
            # Handle virtual-hosted style: bucket-name.s3.region.amazonaws.com
            bucket = parsed.netloc.split(".s3.")[0]
            prefix = parsed.path.lstrip("/")
        else:
            raise ValueError(
                f"Invalid S3 URL format: {s3_url}. "
                "Expected s3://bucket/path or https://bucket.s3.region.amazonaws.com/path"
            )

        # Ensure prefix ends with / for directory listing (unless empty)
        if prefix and not prefix.endswith("/"):
            prefix += "/"

        return bucket, prefix

    def list_video_files(self, s3_url: str) -> List[dict]:
        """
        List all .mp4 video files in the specified S3 directory.

        Args:
            s3_url: S3 URL to the directory containing videos

        Returns:
            List of dicts with 'key', 'url', and 'name' for each video
        """
        bucket, prefix = self.parse_s3_url(s3_url)
        videos = []

        paginator = self.s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

        for page in pages:
            if "Contents" not in page:
                continue

            for obj in page["Contents"]:
                key = obj["Key"]
                if key.lower().endswith(".mp4"):
                    # Generate presigned URL for the video
                    video_url = self.s3_client.generate_presigned_url(
                        "get_object",
                        Params={"Bucket": bucket, "Key": key},
                        ExpiresIn=3600 * 6,  # 6 hours validity
                    )

                    # Extract filename without extension
                    filename = os.path.basename(key)
                    name_without_ext = os.path.splitext(filename)[0]

                    videos.append(
                        {
                            "key": key,
                            "url": video_url,
                            "name": name_without_ext,
                            "bucket": bucket,
                        }
                    )

        return videos

    def upload_caption(
        self, bucket: str, video_key: str, caption: str
    ) -> str:
        """
        Upload a caption file to S3 in the same directory as the video.

        Args:
            bucket: S3 bucket name
            video_key: Original video file key
            caption: Caption text content

        Returns:
            S3 key of the uploaded caption file
        """
        # Replace .mp4 extension with .txt
        caption_key = re.sub(r"\.mp4$", ".txt", video_key, flags=re.IGNORECASE)

        self.s3_client.put_object(
            Bucket=bucket,
            Key=caption_key,
            Body=caption.encode("utf-8"),
            ContentType="text/plain; charset=utf-8",
        )

        return caption_key

    def check_caption_exists(self, bucket: str, video_key: str) -> bool:
        """
        Check if a caption file already exists for a video.

        Args:
            bucket: S3 bucket name
            video_key: Original video file key

        Returns:
            True if caption exists, False otherwise
        """
        caption_key = re.sub(r"\.mp4$", ".txt", video_key, flags=re.IGNORECASE)

        try:
            self.s3_client.head_object(Bucket=bucket, Key=caption_key)
            return True
        except self.s3_client.exceptions.ClientError:
            return False
