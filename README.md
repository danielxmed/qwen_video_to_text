# Video to Text Captioning with Qwen2.5-VL

Generate captions for videos stored in AWS S3 using the Qwen2.5-VL-7B-Instruct model. Supports multi-GPU parallel processing for high throughput.

## Features

- Process videos directly from S3 URLs (no local download required)
- Multi-GPU parallel processing
- Automatic caption upload back to S3
- Skip already processed videos
- Configurable FPS sampling and token limits

## Requirements

- Python 3.10+
- CUDA-capable GPU(s) with at least 16GB VRAM each
- AWS S3 access credentials

## Installation

```bash
pip install -r requirements.txt
```

For Flash Attention 2 support (recommended for faster inference):
```bash
pip install flash-attn --no-build-isolation
```

## Configuration

Copy `.env.example` to `.env` and fill in your AWS credentials:

```bash
cp .env.example .env
```

Edit `.env`:
```
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1
```

## Usage

Basic usage:
```bash
python main.py s3://your-bucket/videos/
```

With options:
```bash
python main.py s3://your-bucket/videos/ \
    --fps 2.0 \
    --max-tokens 1024 \
    --num-gpus 2 \
    --prompt "Describe this video in detail"
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `s3_url` | S3 URL to directory with videos | Required |
| `--prompt` | Custom prompt for generation | Built-in detailed prompt |
| `--fps` | Frames per second to sample | 1.0 |
| `--max-tokens` | Max tokens per caption | 512 |
| `--num-gpus` | Number of GPUs to use | All available |
| `--no-skip-existing` | Process even if caption exists | False |

## Output

For each `video.mp4` in the S3 directory, a corresponding `video.txt` file will be created in the same location containing the generated caption.

## License

MIT
