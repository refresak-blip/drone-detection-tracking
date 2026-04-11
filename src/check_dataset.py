from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_EXTENSION = ".txt"


def count_files(folder: Path, valid_extensions: set[str]) -> int:
    """Count files in a folder with given extensions."""
    if not folder.exists() or not folder.is_dir():
        return 0

    return sum(
        1 for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() in valid_extensions
    )


def check_split(dataset_root: Path, split_name: str) -> None:
    """Check one dataset split (train / valid / test)."""
    split_path = dataset_root / split_name
    images_path = split_path / "images"
    labels_path = split_path / "labels"

    print(f"\n--- {split_name.upper()} ---")
    print(f"Split path exists: {split_path.exists()}")
    print(f"Images path exists: {images_path.exists()}")
    print(f"Labels path exists: {labels_path.exists()}")

    image_count = count_files(images_path, IMAGE_EXTENSIONS)
    label_count = count_files(labels_path, {LABEL_EXTENSION})

    print(f"Number of images: {image_count}")
    print(f"Number of labels: {label_count}")

    if image_count == label_count:
        print("Status: OK (images and labels count match)")
    else:
        print("Status: WARNING (images and labels count do not match)")


def main():
    dataset_root = Path("data/processed")

    print("=" * 50)
    print("DATASET CHECK")
    print("=" * 50)
    print(f"Dataset root: {dataset_root.resolve()}")

    if not dataset_root.exists():
        print("ERROR: data/processed folder does not exist.")
        return

    yaml_path = dataset_root / "data.yaml"
    print(f"\ndata.yaml exists: {yaml_path.exists()}")

    for split in ["train", "valid", "test"]:
        check_split(dataset_root, split)

    print("\nDataset check completed.")


if __name__ == "__main__":
    main()
