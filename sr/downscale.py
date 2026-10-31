import argparse
from pathlib import Path
from PIL import Image


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def resize_images_in_folder(folder_path: str, n: float) -> None:
    input_dir = Path(folder_path)

    if not input_dir.exists():
        raise FileNotFoundError(f"folder not found: {folder_path}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"not a folder: {folder_path}")
    if n <= 0:
        raise ValueError("n must be positive.")

    output_dir = input_dir.parent / f"{input_dir.name}_x{n:g}"
    output_dir.mkdir(exist_ok=True)

    image_files = [
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not image_files:
        print("no images to process.")
        return

    for file_path in image_files:
        try:
            with Image.open(file_path) as img:
                # convert mode so that it can be saved as PNG
                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGBA")

                new_width = max(1, int(img.width / n))
                new_height = max(1, int(img.height / n))

                resized_img = img.resize((new_width, new_height), Image.LANCZOS)

                output_file = output_dir / f"{file_path.stem}.png"
                resized_img.save(output_file, format="PNG")

                print(f"saved: {output_file}")

        except Exception as e:
            print(f"failed: {file_path.name} -> {e}")

    print(f"\ndone. output folder: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Downscales every image in a folder by 1/n (Lanczos) and saves it as PNG to <folder>_x<n>."
    )
    parser.add_argument(
        "folder_path",
        type=str,
        help="input image folder"
    )
    parser.add_argument(
        "n",
        type=float,
        help="downscale factor n (e.g. 2 halves width and height)"
    )

    args = parser.parse_args()
    resize_images_in_folder(args.folder_path, args.n)


if __name__ == "__main__":
    main()