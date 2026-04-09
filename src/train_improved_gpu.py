from ultralytics import YOLO


def main():
    print("Starting improved GPU training with YOLO11s...")

    
    model = YOLO("yolo11s.pt")


    model.train(
        data="data/processed/data.yaml",   
        epochs=30,                       
        imgsz=640,                         
        batch=8,                           
        device=0,                         
        workers=4,                        
        name="improved_yolo11s_gpu",       
        project="runs/detect",            
        pretrained=True,                  
        val=True,                         
        verbose=True                      
    )

    print("Improved GPU training finished!")


if __name__ == "__main__":
    main()