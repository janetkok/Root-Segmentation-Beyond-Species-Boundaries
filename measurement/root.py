import cv2
import numpy as np
import os
import math
import glob
import csv
from natsort import natsorted
def calculate_radial_sweep_radius(contour, centroid):
    """
    Performs a 360-degree radial sweep to calculate average radius.
    """
    cX, cY = centroid
    
    # 1. Convert all contour points to Polar Coordinates relative to centroid
    # Contour shape is (N, 1, 2) -> (N, 2)
    points = contour.reshape(-1, 2)
    
    # Calculate dx, dy
    dx = points[:, 0] - cX
    dy = points[:, 1] - cY
    
    # Calculate r (distance) and theta (angle in degrees)
    r = np.sqrt(dx**2 + dy**2)
    theta = np.degrees(np.arctan2(dy, dx))
    
    # Normalize theta to 0-360 range (arctan2 returns -180 to 180)
    theta = np.mod(theta, 360)
    
    # 2. Bin into 360 degrees
    # We create an array to store the radius found at each degree
    # Initialize with NaNs
    radii_at_degrees = np.full(360, np.nan)
    
    # Round angles to nearest integer degree
    angles_int = np.round(theta).astype(int) % 360
    
    # For each unique angle found, store the mean radius at that angle
    unique_angles = np.unique(angles_int)
    for angle in unique_angles:
        radii_at_degrees[angle] = np.mean(r[angles_int == angle])
        
    # 3. Interpolate missing angles
    # Get indices of valid (non-NaN) data
    valid_idx = np.where(~np.isnan(radii_at_degrees))[0]
    
    if len(valid_idx) == 0:
        return 0, [] 

    # Interpolate
    radii_filled = np.interp(range(360), valid_idx, radii_at_degrees[valid_idx])
    
    # 4. Calculate Final Average
    avg_radius = np.mean(radii_filled)
    
    return avg_radius, radii_filled

def calculate_root_diameter(image_path):
    """
    Calculates the average diameter of the entire root structure.
    Logic: Includes ALL colors except the background (black).
    """
    # 1. Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return

    # 2. Create Mask for "Everything except Background"
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold: Anything brighter than 0 (black) is considered root tissue
    # This captures magenta, orange, blue, yellow, etc. all at once.
    _, mask_root = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # 3. Clean up the mask
    # Close small internal holes or gaps in the segmentation
    kernel = np.ones((9, 9), np.uint8)
    mask_root = cv2.morphologyEx(mask_root, cv2.MORPH_CLOSE, kernel)

    # 4. Determine Outer Boundary (Convex Hull)
    # Find contours of the entire root mass
    contours, _ = cv2.findContours(mask_root, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print(f"[{image_path}] Error: No root tissue detected.")
        return

    # Filter noise: keep only the largest structure
    largest_contour = max(contours, key=cv2.contourArea)
    max_area = cv2.contourArea(largest_contour)

    # Filter contours based on relative size.
    # We only keep contours that are at least 10% (0.05) the size of the main root.
    # This keeps valid large chunks if the cortex is broken, but removes small detached lumps/debris.
    valid_contours = [c for c in contours if cv2.contourArea(c) > (0.05 * max_area)]
    
    # Combine all points from all valid chunks
    all_points = np.vstack(valid_contours)
    
    
    # Calculate Convex Hull
    # This acts like a rubber band around the root, bridging any gaps in the outer epidermis
    root_hull = cv2.convexHull(all_points)
    
    # 5. Calculate Centroid
    M = cv2.moments(root_hull)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = img.shape[1]//2, img.shape[0]//2

    # 6. Calculate Radius and Diameter
    # We calculate the average radius from the centroid to the Hull
    avg_radius, radial_profile = calculate_radial_sweep_radius(root_hull, (cX, cY))
    
    avg_diameter = avg_radius * 2
    
    # Output Results
    print(f"--- Processing {image_path} ---")
    print(f"Method: Whole Root (Non-Black Pixels)")
    print(f"Average Radius:   {avg_radius:.2f} pixels")
    print(f"Average Diameter: {avg_diameter:.2f} pixels")
    
    # --- Visualization ---
    viz_img = img.copy()
    
    # Draw the calculated Hull (Green) - The boundary used for measurement
    cv2.drawContours(viz_img, [root_hull], -1, (0, 255, 0), 3)
    
    # Draw Radial Rays (Every 15 degrees) to show measurement
    for angle in range(0, 360, 15):
        r = radial_profile[angle]
        rad_angle = np.radians(angle)
        end_x = int(cX + r * np.cos(rad_angle))
        end_y = int(cY + r * np.sin(rad_angle))
        cv2.line(viz_img, (cX, cY), (end_x, end_y), (0, 255, 255), 1)

    # Draw Center
    cv2.circle(viz_img, (cX, cY), 5, (0, 0, 255), -1) 
    
    # Save output
    output_filename = f"diameter_measured_{os.path.basename(image_path)}"
    cv2.imwrite(output_filename, viz_img)
    print(f"Saved visualization to: {output_filename}\n")
    return avg_diameter

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
            diameter = calculate_root_diameter(img_path)
            
            if diameter is not None:
                results.append([filename, f"{diameter:.2f}"])
                print(f"Diameter: {diameter:.2f}")
            else:
                results.append([filename, "N/A"])
                print("Failed.")
                
        except Exception as e:
            print(f"Error: {e}")
            results.append([filename, "Error"])

    # Save to CSV
    csv_path = os.path.join(folder_path, "root_diameter_results.csv")
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "Stele Diameter (Pixels)"])
        writer.writerows(results)
        
    print(f"\nProcessing complete.")
    print(f"Results saved to: {csv_path}")

# Main execution block
if __name__ == "__main__":
    # --- CONFIGURATION ---
    target_folder = r'D:/plant_research/measurement/Millet_For_Measurement/predict'
    # ---------------------
    
    if os.path.exists(target_folder):
        process_directory(target_folder)
    else:
        print(f"Error: Directory not found: {target_folder}")