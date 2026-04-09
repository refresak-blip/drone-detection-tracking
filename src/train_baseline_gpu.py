from ultralytics import YOLO


def main():
    print("Starting baseline GPU training...")

 
    model = YOLO("yolo11n.pt")

   
    model.train(
        data="data/processed/data.yaml",   
        epochs=30,                   
        imgsz=640,                  
        batch=16,                     
        device=0,                     
        workers=4,                        
        name="baseline_yolo11n_gpu",       
        project="runs/detect",             
        pretrained=True,                 
        val=True,                          
        verbose=True                       
    )

    print("Baseline GPU training finished!")