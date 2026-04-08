import random
import shutil
from pathlib import Path


RAW_DATASET = Path("data/raw/dataset")
PROCESSED_DATASET = Path("data/processed")

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


images_dir = RAW_DATASET / "images"
labels_dir = RAW_DATASET / "labels"


def copy_files(file_list, split_name):
    dest_img_dir = PROCESSED_DATASET / split_name / "images"
    dest_lbl_dir = PROCESSED_DATASET / split_name / "labels"

    dest_img_dir.mkdir(parents=True, exist_ok=True)
    dest_lbl_dir.mkdir(parents=True, exist_ok=True)

    copied_labels = 0
    missing_labels = 0

    for img_path in file_list:
        label_path = labels_dir / f"{img_path.stem}.txt"

        shutil.copy2(img_path, dest_img_dir / img_path.name)

        if label_path.exists():
            shutil.copy2(label_path, dest_lbl_dir / label_path.name)
            copied_labels += 1
        else:
            missing_labels += 1

    print(f"\n[{split_name}]")
    print(f"Copied images: {len(file_list)}")
    print(f"Copied labels: {copied_labels}")
    print(f"Missing labels: {missing_labels}")


def main():
    print("=" * 50)
    print("PREPARING DATASET SPLIT")
    print("=" * 50)

    if not images_dir.exists():
        print(f"ERROR: Images folder not found: {images_dir}")
        return

    if not labels_dir.exists():
        print(f"ERROR: Labels folder not found: {labels_dir}")
        return

    image_files = [
        f for f in images_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if len(image_files) == 0:
        print("ERROR: No image files found.")
        return

    random.shuffle(image_files)

    total = len(image_files)
    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    train_files = image_files[:train_count]
    val_files = image_files[train_count:train_count + val_count]
    test_files = image_files[train_count + val_count:]

    print(f"Total images: {total}")
    print(f"Train: {len(train_files)}")
    print(f"Valid: {len(val_files)}")
    print(f"Test: {len(test_files)}")

    copy_files(train_files, "train")
    copy_files(val_files, "valid")
    copy_files(test_files, "test")

    print("\nDataset split completed successfully!")


if __name__ == "__main__":
    main()