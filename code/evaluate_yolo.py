import os
import torch
from ultralytics import YOLO


def main():
  
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_yaml = os.path.join(project_root, "data", "data.yaml")

   
    runs_dir = os.path.join(project_root, "runs")
    run_name = "pcb_yolo11n"
    weights_path = os.path.join(runs_dir, run_name, "weights", "best.pt")

    print("Project root:", project_root)
    print("Data yaml:", data_yaml)
    print("Weights path:", weights_path)

    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"data.yaml not found at {data_yaml}")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Could not find best weights at {weights_path}. "
            "Make sure training finished successfully."
        )



    device = 0 if torch.cuda.is_available() else "cpu"
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Using device:", device)

  
    model = YOLO(weights_path)

  
    metrics = model.val(
        data=data_yaml,
        imgsz=640,
        device=device,
        split="val", 
        project=os.path.join(project_root, "runs"),
        name="pcb_yolo11n_val",
        workers=0
    )

    print("\nValidation complete.")
    print("Results dir:", metrics.save_dir)
    print("Metrics dict:")
    print(metrics.results_dict)


if __name__ == "__main__":
    main()
