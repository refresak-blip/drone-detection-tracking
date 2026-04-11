from pathlib import Path
from ultralytics import YOLO


def main():
    print("=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)

    model_path = Path("runs/detect/runs/detect/improved_yolo11s_gpu/weights/best.pt")
    data_yaml = Path("data/processed/data.yaml")

    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        return

    if not data_yaml.exists():
        print(f"ERROR: data.yaml not found: {data_yaml}")
        return

    print(f"Using model: {model_path}")
    print(f"Using dataset config: {data_yaml}")

    model = YOLO(str(model_path))

    results = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=640,
        batch=8,
        device=0
    )

    print("\nEvaluation completed.")
    print(f"Precision: {results.box.mp}")
    print(f"Recall: {results.box.mr}")
    print(f"mAP50: {results.box.map50}")
    print(f"mAP50-95: {results.box.map}")


if __name__ == "__main__":
    main()
