from pathlib import Path
from ultralytics import YOLO


def main():
    print("Starting live camera tracking...")

    
    model_path = "runs/detect/runs/detect/improved_yolo11s_gpu/weights/best.pt"

    
    camera_source = 0

    if not Path(model_path).exists():
        print(f"ERROR: Model not found: {model_path}")
        return

    model = YOLO(model_path)

    model.track(
        source=camera_source,
        show=True,
        save=False,
        tracker="bytetrack.yaml",
        conf=0.7,
        iou=0.4,
        imgsz=640,
        device=0,
        persist=True
    )

    print("Camera trracking finished.")


if __name__ == "__main__":
    main()