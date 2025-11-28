import os
import cv2
import numpy as np


def main():
  
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    images_dir = os.path.join(project_root, "images")

    image_name = "motherboard_image.JPEG"   
    image_path = os.path.join(images_dir, image_name)

    print("Files in images_dir:", os.listdir(images_dir))

    output_dir = os.path.join(images_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    print("Project root:", project_root)
    print("Input image:", image_path)
    print("Output dir:", output_dir)


    #  Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    original = img.copy()

    # Grayscale + blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)


    # Threshold (Otsu) 


    # Normal Otsu threshold 
    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Inverted version
    _, thresh_inv = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )


    # Edge detection 
    edges = cv2.Canny(blurred, 50, 150)


    # contours on the INVERTED threshold 
    contours, _ = cv2.findContours(
        thresh_inv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise RuntimeError("No contours found on threshold image.")

 
    largest_contour = max(contours, key=cv2.contourArea)


    # Create mask from contour
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)


    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)


    #Apply mask to original image
    pcb_extracted = cv2.bitwise_and(original, original, mask=mask_clean)

  
    cv2.imwrite(os.path.join(output_dir, "01_gray.png"), gray)
    cv2.imwrite(os.path.join(output_dir, "02_blurred.png"), blurred)
    cv2.imwrite(os.path.join(output_dir, "03_threshold.png"), thresh)
    cv2.imwrite(os.path.join(output_dir, "03b_threshold_inv.png"), thresh_inv)
    cv2.imwrite(os.path.join(output_dir, "04_edges.png"), edges)
    cv2.imwrite(os.path.join(output_dir, "05_mask_raw.png"), mask)
    cv2.imwrite(os.path.join(output_dir, "06_mask_clean.png"), mask_clean)
    cv2.imwrite(os.path.join(output_dir, "07_pcb_extracted.png"), pcb_extracted)

    print("Saved key images to:", output_dir)
    print(" - 04_edges.png (edge detection)")
    print(" - 05_mask_raw.png / 06_mask_clean.png (mask)")
    print(" - 07_pcb_extracted.png (final extracted motherboard)")


if __name__ == "__main__":
    main()
