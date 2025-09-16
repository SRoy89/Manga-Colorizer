import argparse
import os
import numpy as np
import PIL.Image as Image
import torch

from denoisator import MangaDenoiser
from colorizator import MangaColorizator
from upscalator import MangaUpscaler
from utils.utils import distance_from_grayscale, save_image, clear_torch_cache


def make_divisible_by_32(n):
    """Round up to nearest multiple of 32"""
    return (n + 31) // 32 * 32


def process_image(image_path, output_folder, colorizer, upscaler, denoiser, config):
    image_name = os.path.basename(image_path)
    pil_img = Image.open(image_path).convert("RGB")
    image = np.array(pil_img)

    # Skip already colored images
    coloredness = distance_from_grayscale(image)
    if coloredness > 1:
        print(f"[+] {image_name} is already colored, skipping.")
        return

    # Step 1: Denoise
    if config.denoise:
        print(f"[*] Denoising {image_name}...")
        image = denoiser.denoise(image, config.denoise_sigma)

    # Step 2: Colorize
    if config.colorize:
        if config.colorized_image_size is None:
            img_width = pil_img.width
            adjusted_width = make_divisible_by_32(img_width)
            if adjusted_width != img_width:
                print(f"[*] Auto-detected width for {image_name}: {img_width} -> adjusted to {adjusted_width}")
            else:
                print(f"[*] Auto-detected width for {image_name}: {img_width} (already divisible by 32)")
            colorizer.set_image((image.astype('float32') / 255), adjusted_width)
        else:
            adjusted_width = make_divisible_by_32(config.colorized_image_size)
            print(f"[*] Using provided width: {config.colorized_image_size} -> adjusted to {adjusted_width}")
            colorizer.set_image((image.astype('float32') / 255), adjusted_width)

        print(f"[*] Colorizing {image_name}...")
        image = colorizer.colorize()

    # Step 3: Upscale
    if config.upscale:
        print(f"[*] Upscaling {image_name} by {config.upscale_factor}x...")
        image = upscaler.upscale((image.astype('float32') / 255), config.upscale_factor)

    # Save
    output_path = os.path.join(output_folder, image_name)
    save_image(image, output_path)
    print(f"[+] Processed {image_name} -> Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch Colorize Images")
    parser.add_argument("--input_path", type=str, default="input", help="Folder containing images")
    parser.add_argument("--output_path", type=str, default="output", help="Folder to save processed images")
    parser.add_argument("--colorized_image_size", type=int, default=None,
                        help="Target width for colorization. If not provided, auto-detects width of each image.")

    # Auto-detect GPU availability
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=default_device, help='Device to use')

    parser.add_argument('--colorizer_path', default='networks/generator.zip')
    parser.add_argument('--extractor_path', default='networks/extractor.pth')
    parser.add_argument('--upscaler_path', default='networks/RealESRGAN_x4plus_anime_6B.pt')
    parser.add_argument('--upscaler_type', choices=['ESRGAN', 'GigaGAN'], default='ESRGAN')

    parser.add_argument('--no-upscale', dest='upscale', action='store_false', default=True, help='Disable upscaling')
    parser.add_argument('--no-colorize', dest='colorize', action='store_false', default=True, help='Disable colorization')
    parser.add_argument('--no-denoise', dest='denoise', action='store_false', default=True, help='Disable denoiser')

    parser.add_argument('--upscale_factor', choices=[2, 4], default=4, type=int, help='Upscale by x2 or x4')
    parser.add_argument('--denoise_sigma', default=25, type=int, help='How much noise to expect from the image')

    config = parser.parse_args()
    os.makedirs(config.output_path, exist_ok=True)

    # Default tile settings
    config.upscaler_tile_size = 256
    config.colorizer_tile_size = 0
    config.tile_pad = 8

    # Initialize components
    colorizer = MangaColorizator(config) if config.colorize else None
    upscaler = MangaUpscaler(config) if config.upscale else None
    denoiser = MangaDenoiser(config) if config.denoise else None
    print(f"[+] Components initialized on device: {config.device}")

    # Collect images
    images = [f for f in os.listdir(config.input_path) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
    for img in images:
        process_image(os.path.join(config.input_path, img), config.output_path, colorizer, upscaler, denoiser, config)

    print("[+] Batch processing complete")

    clear_torch_cache()
    print("[+] Components released")


if __name__ == "__main__":
    main()
