from ultralytics import YOLO


def main():
    print("Starting quick CPU baseline test...")

    model = YOLO("yolo11n.pt")

    model.train(
        data="data/processed/data.yaml",
        epochs=1,
        imgsz=416,
        batch=4,
        name="baseline_yolo11n_cpu_test",
        device="cpu"
    )

    print("Quick test finished!")


if __name__ == "__main__":
    main()