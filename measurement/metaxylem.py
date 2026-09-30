import cv2
import numpy as np
import glob
import os
import csv
from natsort import natsorted
def count_metaxylem(image_path):
    # 1. Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # 2. Define the target color in BGR
    # User requested [0, 0, 255] (Red).
    # NOTE: If you intended to count the large BLUE vessels (RGB 0,0,255), 
    # change the values below to: target_bgr = np.array([255, 0, 0])
    target_bgr = np.array([255, 0, 0]) 

    # 3. Define a sensitivity range (to handle JPEG artifacts)
    sensitivity = 40
    lower_bound = np.clip(target_bgr - sensitivity, 0, 255)
    upper_bound = np.clip(target_bgr + sensitivity, 0, 255)

    # 4. Create a binary mask for the target color
    mask = cv2.inRange(img, lower_bound, upper_bound)

    # 5. Clean up the mask (optional morphological operations)
    # Remove small noise pixels
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 6. Find contours (connected components)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 7. Count valid metaxylem regions
    # Filter out tiny noise contours if necessary (e.g., area < 10 pixels)
    min_area = 4000
    metaxylem_count = 0
    
    # Create a copy of the image to visualize results
    result_img = img.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            metaxylem_count += 1
            # Draw green outline around detected metaxylem
    #         cv2.drawContours(result_img, [cnt], -1, (0, 255, 0), 2)

    # # 8. Output results
    # print(f"Processing image: {image_path}")
    # print(f"Target BGR Color: {target_bgr}")
    # print(f"Found {metaxylem_count} metaxylem regions.")

    # # Save the visualization
    # output_filename = 'metaxylem_detected.jpg'
    # cv2.imwrite(output_filename, result_img)
    # print(f"Saved detection visualization to {output_filename}")
    return metaxylem_count

def process_directory(folder_path):
    print(f"Scanning directory: {folder_path}")
    
    # Find all images
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    if not image_files:
        print("No images found in the directory.")
        return

    print(f"Found {len(image_files)} images. Starting processing...")
    
    results = []
    
    for i, img_path in enumerate(natsorted(image_files)):
        filename = os.path.basename(img_path)
        print(f"[{i+1}/{len(image_files)}] Processing {filename}...", end=" ")
        
        try:
            count = count_metaxylem(img_path)
            
            if count is not None:
                results.append([filename, f"{count}"])
                print(f"Count: {count}")
            else:
                results.append([filename, "N/A"])
                print("Failed.")
                
        except Exception as e:
            print(f"Error: {e}")
            results.append([filename, "Error"])

    # Save to CSV
    csv_path = os.path.join(folder_path, "metaxylem_count_results.csv")
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "Metaxylem Count"])
        writer.writerows(results)
        
    print(f"\nProcessing complete.")
    print(f"Results saved to: {csv_path}")
    
# Main execution block
if __name__ == "__main__":
    # --- CONFIGURATION ---
    target_folder = r'D:/plant_research/measurement/Sorghum_For_Measurement/predict'
    # ---------------------
    
    if os.path.exists(target_folder):
        process_directory(target_folder)
    else:
        print(f"Error: Directory not found: {target_folder}")
