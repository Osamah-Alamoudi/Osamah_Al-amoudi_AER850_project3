import os
import torch
from ultralytics import YOLO


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # trained weights from run
    runs_dir = os.path.join(project_root, "runs")
    run_name = "pcb_yolo11n"
    weights_path = os.path.join(runs_dir, run_name, "weights", "best.pt")

    # evaluation image directory
    eval_images_dir = os.path.join(project_root, "data", "evaluation", "images")

    print("Weights path:", weights_path)
    print("Eval images dir:", eval_images_dir)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"best.pt not found at {weights_path}")
    if not os.path.isdir(eval_images_dir):
        raise FileNotFoundError(f"Evaluation images folder not found at {eval_images_dir}")

    device = 0 if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = YOLO(weights_path)

    results = model.predict(
        source=eval_images_dir,
        imgsz=640,
        conf=0.25, 
        device=device,
        save=True,
        project=os.path.join(project_root, "runs"),
        name="pcb_yolo11n_eval_pred",
        workers=0,
        line_width=5,  
    )

    print("Predictions saved to:", results[0].save_dir)


if __name__ == "__main__":
    main()

