import os
import torch
from ultralytics import YOLO


def main():
  

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_yaml = os.path.join(project_root, "data", "data.yaml")

    print("Project root:", project_root)
    print("Data yaml:", data_yaml)

    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"data.yaml not found at {data_yaml}")

  
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    device = 0 if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("WARNING: Training will run on CPU.")


    model = YOLO("yolo11n.pt")

 
    results = model.train(
        data=data_yaml,                
        epochs=120,                       
        imgsz=640,                       
        batch=16,                        
        device=device,                    
        project=os.path.join(project_root, "runs"),
        name="pcb_yolo11n",         
        workers=0,
        patience=10,        # early stopping patience
        verbose=True
    )

    print("Training complete.")
    print("Results saved in:", results.save_dir)


if __name__ == "__main__":
    main()
