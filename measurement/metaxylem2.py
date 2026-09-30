import cv2
import numpy as np
import os
import glob
import csv
from natsort import natsorted
def single_stats_filtered(binary_mask):
    """
    Implements the user's specific algorithm:
    1. Find all connected components.
    2. Calculate average area.
    3. Filter out components smaller than 0.3 * Average Area.
    4. Return count and final stats.
    """
    # Get connected components of the raw mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)
    
    # If no components found (except background), return empty
    if num_labels <= 1:
        return 0, 0, np.zeros_like(binary_mask)

    # Calculate the threshold based on the algorithm theory
    # Note: stats[0] is the background, so we skip it (range 1 to num_labels)
    total_area_raw = 0
    for i in range(1, num_labels):
        total_area_raw += stats[i, cv2.CC_STAT_AREA]
    
    # Average area of raw components
    avg_area_raw = total_area_raw / (num_labels - 1)
    
    # Threshold: 30% of the average area
    area_threshold = avg_area_raw * 0.3

    # Filter regions
    filtered_regions = np.zeros_like(binary_mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > area_threshold:
            filtered_regions[labels == i] = 255

    # Re-calculate stats on the filtered regions
    num_labels_final, labels_final, stats_final, _ = cv2.connectedComponentsWithStats(filtered_regions)
    
    # Count (subtract 1 for background)
    final_count = num_labels_final - 1
    
    # Calculate final average area
    final_total_area = 0
    final_avg_area = 0
    if final_count > 0:
        for i in range(1, num_labels_final):
            final_total_area += stats_final[i, cv2.CC_STAT_AREA]
        final_avg_area = final_total_area / final_count

    return final_count, final_avg_area, filtered_regions

def process_metaxylem_count(image_path, save_viz=True):
    # 1. Load Image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading {image_path}")
        return None, None

    # 2. Isolate Metaxylem (Blue)
    # OpenCV uses BGR. Blue is [255, 0, 0].
    # We use a range to capture slight variations in the mask edges.
    lower_meta = np.array([150, 0, 0])
    upper_meta = np.array([255, 100, 100])
    
    # Create Binary Mask (0 or 255)
    mask_meta = cv2.inRange(img, lower_meta, upper_meta)

    # 3. Apply the specific Counting Algorithm
    count, avg_area, filtered_mask = single_stats_filtered(mask_meta)

    # 4. Visualization (Optional)
    
    viz_img = img.copy()
    
    # Find contours of the VALID (filtered) metaxylem to draw them
    contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw outlines in bright yellow and put numbers
    for i, cnt in enumerate(contours):
        cv2.drawContours(viz_img, [cnt], -1, (0, 255, 255), 2)
        
        # Put the number ID
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(viz_img, str(i+1), (cX - 10, cY), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Save visualization
    viz_dir = os.path.join(os.path.dirname(image_path), "metaxylem_counts")
    os.makedirs(viz_dir, exist_ok=True)
    cv2.imwrite(os.path.join(viz_dir, f"counted_{os.path.basename(image_path)}"), viz_img)

    return count, avg_area

def main():
    # --- CONFIGURATION ---
    # Path to your images
    input_folder = r"D:/plant_research/measurement/Millet_For_Measurement/predict"
    # ---------------------

    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
    files = []
    for ext in image_extensions:
        files.extend(glob.glob(os.path.join(input_folder, ext)))

    if not files:
        print("No images found.")
        return

    print(f"Found {len(files)} images. Analyzing Metaxylem...")

    results = []
    for f in natsorted(files):
        filename = os.path.basename(f)
        # Skip previously processed visualization files to avoid duplication
        if "counted_" in filename or "processed_" in filename:
            continue

        print(f"Processing {filename}...", end=" ")
        count, avg_area = process_metaxylem_count(f)
        
        if count is not None:
            print(f"Count: {count}")
            results.append([filename, count, f"{avg_area:.2f}"])
        else:
            print("Failed")
            results.append([filename, "Error", "Error"])

    # Save to CSV
    csv_filename = os.path.join(input_folder, "metaxylem_counts.csv")
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "Metaxylem_Count", "Avg_Area_Pixels"])
        writer.writerows(results)

    print(f"\nDone. Results saved to {csv_filename}")

if __name__ == "__main__":
    main()