from ultralytics import YOLO
from pathlib import Path


def main():
    print("Starting video tracking...")

    
    model_path = "runs/detect/runs/detect/improved_yolo11s_gpu/weights/best.pt"

    source_video = "data/sample_videos/drone_test.mp4"

    
    output_project = r"D:\CV_Projects\drone-detection-tracking\results\videos"
    output_name = "drone_tracking_demo"

    
    if not Path(model_path).exists():
        print(f"ERROR: Model not found: {model_path}")
        return

    if not Path(source_video).exists():
        print(f"ERROR: Video not found: {source_video}")
        return

    model = YOLO(model_path)

    
    model.track(
        source=source_video,
        save=True,
        project=output_project,
        name=output_name,
        tracker="bytetrack.yaml",
        conf=0.7,
        iou=0.3,
        imgsz=640,
        device=0,
        show=False,
        persist=True
    )

    print("Video tracking finished!")


if __name__ == "__main__":
    main()