from ultralytics import YOLO
from pathlib import Path


def main():
    print("Starting video detection...")


    model_path = "runs/detect/runs/detect/improved_yolo11s_gpu/weights/best.pt"

    source_video = "data/sample_videos/drone_test.mp4"

    output_project = "results/videos"
    output_name = "drone_detection_demo"

    
    if not Path(source_video).exists():
        print(f"ERROR: Video not found: {source_video}")
        return

    if not Path(model_path).exists():
        print(f"ERROR: Model not found: {model_path}")
        return

    model = YOLO(model_path)

    model.predict(
        source=source_video,
        save=True,
        project=output_project,
        name=output_name,
        conf=0.6,
        imgsz=640,
        device=0,
        show=False
    )

    print("Video detection finished!")


if __name__ == "__main__":
    main()