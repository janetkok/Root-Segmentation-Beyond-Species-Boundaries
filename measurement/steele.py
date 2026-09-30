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
    points = contour.reshape(-1, 2)
    dx = points[:, 0] - cX
    dy = points[:, 1] - cY
    
    r = np.sqrt(dx**2 + dy**2)
    theta = np.degrees(np.arctan2(dy, dx))
    theta = np.mod(theta, 360)
    
    # 2. Bin into 360 degrees
    radii_at_degrees = np.full(360, np.nan)
    angles_int = np.round(theta).astype(int) % 360
    
    unique_angles = np.unique(angles_int)
    for angle in unique_angles:
        radii_at_degrees[angle] = np.mean(r[angles_int == angle])
        
    # 3. Interpolate missing angles
    valid_idx = np.where(~np.isnan(radii_at_degrees))[0]
    
    if len(valid_idx) == 0:
        return 0, [] 

    radii_filled = np.interp(range(360), valid_idx, radii_at_degrees[valid_idx])
    
    # 4. Calculate Final Average
    avg_radius = np.mean(radii_filled)
    return avg_radius, radii_filled

def calculate_stele_diameter(image_path):
    """
    Calculates the average diameter of the Stele (Central Cylinder) only.
    Includes: Magenta, Blue, Yellow, and Green layers.
    Excludes: Orange (Cortex) and Black (Background).
    """
    # 1. Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return

    # 2. Define Color Ranges (BGR) for Stele Components
    # Note: We use ranges to account for slight anti-aliasing edges
    
    # Magenta (Central Tissue) - BGR: [255, 0, 255]
    lower_magenta = np.array([200, 0, 200])
    upper_magenta = np.array([255, 100, 255])

    # Blue (Metaxylem Vessels) - BGR: [255, 0, 0] (Red channel in BGR is 0, Blue is 255)
    lower_blue = np.array([200, 0, 0])
    upper_blue = np.array([255, 100, 100])

    # Yellow (Vascular Bundles/Ring) - BGR: [0, 255, 255]
    lower_yellow = np.array([0, 200, 200])
    upper_yellow = np.array([100, 255, 255])
    
    # Green (Endodermis Boundary) - BGR: [0, 255, 0] approx
    # Looking at your images, the green line is roughly [0, 255, 0] or similar
    lower_green = np.array([0, 200, 0])
    upper_green = np.array([100, 255, 100])

    # 3. Create Masks
    mask_stele = cv2.inRange(img, lower_magenta, upper_magenta)


    # 5. Clean up the mask (Close gaps)
    kernel = np.ones((9, 9), np.uint8)
    mask_stele = cv2.morphologyEx(mask_stele, cv2.MORPH_CLOSE, kernel)

    # 6. Find Contours
    contours, _ = cv2.findContours(mask_stele, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print(f"[{image_path}] Error: No stele tissue detected (check color ranges).")
        return

    # Keep only the largest contour (The main central cylinder)
    largest_contour = max(contours, key=cv2.contourArea)
    max_area = cv2.contourArea(largest_contour)

    # Filter contours based on relative size.
    # We only keep contours that are at least 10% (0.05) the size of the main root.
    # This keeps valid large chunks if the cortex is broken, but removes small detached lumps/debris.
    valid_contours = [c for c in contours if cv2.contourArea(c) > (0.05 * max_area)]
    
    # Combine all points from all valid chunks
    all_points = np.vstack(valid_contours)
    
    # Calculate Convex Hull
    # This wraps the outer edge of the stele components, ignoring internal holes
    stele_hull = cv2.convexHull(all_points)
    
    # 7. Calculate Centroid
    M = cv2.moments(stele_hull)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = img.shape[1]//2, img.shape[0]//2

    # 8. Calculate Radius and Diameter
    avg_radius, radial_profile = calculate_radial_sweep_radius(stele_hull, (cX, cY))
    avg_diameter = avg_radius * 2
    
    # Output Results
    print(f"--- Processing {image_path} ---")
    print(f"Target: Stele Only (Magenta/Blue/Yellow/Green)")
    print(f"Average Radius:   {avg_radius:.2f} pixels")
    print(f"Average Diameter: {avg_diameter:.2f} pixels")
    
    # --- Visualization ---
    viz_img = img.copy()
    
    # Highlight the Stele region found
    # Draw the Hull (Cyan)
    cv2.drawContours(viz_img, [stele_hull], -1, (255, 255, 0), 3)
    
    # Draw Radial Rays (Every 20 degrees)
    for angle in range(0, 360, 20):
        r = radial_profile[angle]
        rad_angle = np.radians(angle)
        end_x = int(cX + r * np.cos(rad_angle))
        end_y = int(cY + r * np.sin(rad_angle))
        cv2.line(viz_img, (cX, cY), (end_x, end_y), (0, 0, 255), 5)

    # Center dot
    cv2.circle(viz_img, (cX, cY), 5, (0, 0, 255), -1) 
    
    output_filename = f"stele_diameter_{os.path.basename(image_path)}"
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
            diameter = calculate_stele_diameter(img_path)
            
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
    csv_path = os.path.join(folder_path, "stele_diameter_results.csv")
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