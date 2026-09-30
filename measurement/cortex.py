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
    # (Handles cases where multiple pixels map to the same degree)
    unique_angles = np.unique(angles_int)
    for angle in unique_angles:
        radii_at_degrees[angle] = np.mean(r[angles_int == angle])
        
    # 3. Interpolate missing angles
    # Sometimes (rarely in high res), a degree might be skipped by the pixels.
    # We fill NaNs using linear interpolation.
    
    # Get indices of valid (non-NaN) data
    valid_idx = np.where(~np.isnan(radii_at_degrees))[0]
    
    if len(valid_idx) == 0:
        return 0, [] # Should not happen for valid contours

    # Interpolate
    radii_filled = np.interp(range(360), valid_idx, radii_at_degrees[valid_idx])
    
    # 4. Calculate Final Average
    avg_radius = np.mean(radii_filled)
    
    return avg_radius, radii_filled

def calculate_combined_layer_width(image_path):
    """
    Calculates the average width of the combined layers using Convex Hull 
    to handle missing parts (gaps) in the root structure.
    """
    # 1. Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return None

    # 2. Define color ranges (BGR Format)
    
    # --- Inner Core (Stele Complex) ---
    # Stele (Magenta): BGR [255, 0, 255]
    lower_stele = np.array([200, 0, 200])
    upper_stele = np.array([255, 100, 255])

    # Metaxylem (Blue): RGB [0, 0, 255] -> BGR [255, 0, 0]
    lower_meta = np.array([200, 0, 0])
    upper_meta = np.array([255, 100, 100])

    # Vascular Bundle (Yellow): RGB [255, 255, 0] -> BGR [0, 255, 255]
    lower_vb = np.array([0, 200, 200])
    upper_vb = np.array([100, 255, 255])
    
    # --- Outer Layers for Width Calculation ---
    # Cortex (Orange): BGR [0, 156, 255]
    lower_cortex = np.array([0, 100, 200])
    upper_cortex = np.array([100, 200, 255])
    
    # Sclerenchyma: BGR [152, 60, 125]
    lower_sclera = np.array([120, 30, 95])
    upper_sclera = np.array([185, 90, 155])

    # Epidermis: BGR [127, 3, 255]
    lower_epi = np.array([90, 0, 220])
    upper_epi = np.array([160, 60, 255])

    # Endodermis: BGR [69, 252, 3]
    lower_endo = np.array([30, 200, 0])
    upper_endo = np.array([100, 255, 100])
    
    # Aerenchyma: BGR [0, 0, 255]
    lower_aeren = np.array([0, 0, 200])
    upper_aeren = np.array([60, 60, 255])

    # 3. Create binary masks
    # Inner Core components
    mask_stele_main = cv2.inRange(img, lower_stele, upper_stele)
    mask_meta = cv2.inRange(img, lower_meta, upper_meta)
    mask_vb = cv2.inRange(img, lower_vb, upper_vb)
    mask_endo = cv2.inRange(img, lower_endo, upper_endo)

    # Combine Inner Core components
    mask_stele_combined = cv2.bitwise_or(mask_stele_main, mask_meta)
    mask_stele_combined = cv2.bitwise_or(mask_stele_combined, mask_vb)
    mask_stele_combined = cv2.bitwise_or(mask_stele_combined, mask_endo)
    
    # Outer Layers components
    mask_cortex = cv2.inRange(img, lower_cortex, upper_cortex)
    mask_sclera = cv2.inRange(img, lower_sclera, upper_sclera)
    mask_epi = cv2.inRange(img, lower_epi, upper_epi)
    
    mask_aeren = cv2.inRange(img, lower_aeren, upper_aeren)

    # 4. Combine outer layer masks
    mask_layers = cv2.bitwise_or(mask_cortex, mask_sclera)
    # mask_layers = cv2.bitwise_or(mask_layers, mask_epi)
    # mask_layers = cv2.bitwise_or(mask_layers, mask_endo)
    mask_layers = cv2.bitwise_or(mask_layers, mask_aeren)

    # Clean up masks
    kernel = np.ones((9, 9), np.uint8)
    mask_layers = cv2.morphologyEx(mask_layers, cv2.MORPH_CLOSE, kernel)
    mask_stele_combined = cv2.morphologyEx(mask_stele_combined, cv2.MORPH_CLOSE, kernel)

    # 5. Determine Inner Boundary (Stele Combined)
    contours_stele, _ = cv2.findContours(mask_stele_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours_stele:
        print(f"[{image_path}] Error: Stele (Inner Core) not detected.")
        return None

    inner_contour = max(contours_stele, key=cv2.contourArea)
    
    # Calculate Centroid from Stele (Most reliable reference point)
    M = cv2.moments(inner_contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = img.shape[1]//2, img.shape[0]//2

    # 6. Determine Outer Boundary (Convex Hull) with Noise Filtering
    # Find all contours in the layers
    contours_layers, _ = cv2.findContours(mask_layers, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours_layers:
        print(f"[{image_path}] Error: No layer tissue detected.")
        return None

    # --- UPDATED LOGIC HERE ---
    # Find the largest contour (Main Root Body)
    max_contour = max(contours_layers, key=cv2.contourArea)
    max_area = cv2.contourArea(max_contour)

    # Filter contours based on relative size.
    # We only keep contours that are at least 10% (0.05) the size of the main root.
    # This keeps valid large chunks if the cortex is broken, but removes small detached lumps/debris.
    valid_contours = [c for c in contours_layers if cv2.contourArea(c) > (0.01 * max_area)]
    
    # Combine all points from all valid chunks
    all_points = np.vstack(valid_contours)
    
    # Calculate Convex Hull -> This becomes our Outer Boundary
    outer_contour_hull = cv2.convexHull(all_points)
    
    # 7. Calculate Radii
    r_outer, outer_profile = calculate_radial_sweep_radius(outer_contour_hull, (cX, cY))
    r_inner, inner_profile = calculate_radial_sweep_radius(inner_contour, (cX, cY))
    
    avg_width = r_outer - r_inner
    
    # Output Results
    print(f"--- Processing {image_path} ---")
    print(f"Method: Convex Hull (Filtered Outliers)")
    print(f"Outer Radius (Hull): {r_outer:.2f} pixels")
    print(f"Inner Radius (Stele + Meta + VB): {r_inner:.2f} pixels")
    print(f"Average Combined Width: {avg_width:.2f} pixels")
    
    # --- Visualization ---
    viz_img = img.copy()
    
    # Draw Inner Contour (Solid Red)
    cv2.drawContours(viz_img, [inner_contour], -1, (0, 0, 255), 3)
    
    # Draw Outer Hull (Solid Green)
    cv2.drawContours(viz_img, [outer_contour_hull], -1, (0, 255, 0), 3)
    
    # Draw Radial Rays (Every 10 degrees) based on the Hull
    for angle in range(0, 360, 10):
        r = outer_profile[angle]
        rad_angle = np.radians(angle)
        end_x = int(cX + r * np.cos(rad_angle))
        end_y = int(cY + r * np.sin(rad_angle))
        # Draw ray
        cv2.line(viz_img, (cX, cY), (end_x, end_y), (0, 255, 255), 1)

    # Circles for Average Radii
    cv2.circle(viz_img, (cX, cY), int(r_outer), (0, 255, 0), 2) # Outer Avg
    cv2.circle(viz_img, (cX, cY), int(r_inner), (0, 0, 255), 2) # Inner Avg
    
    cv2.circle(viz_img, (cX, cY), 5, (0, 255, 255), -1) # Center
    
    output_filename = f"processed_hull_{os.path.basename(image_path)}"
    
    # Save visualization to the same folder as the image
    output_path = os.path.join(output_filename)
    cv2.imwrite(output_path, viz_img)
    print(f"Saved visualization to: {output_filename}\n")
    
    return avg_width

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
    
    # Sort files naturally
    try:
        sorted_files = natsorted(image_files)
    except Exception:
        sorted_files = sorted(image_files)

    for i, img_path in enumerate(sorted_files):
        filename = os.path.basename(img_path)
        print(f"[{i+1}/{len(image_files)}] Processing {filename}...", end=" ")
        
        try:
            diameter = calculate_combined_layer_width(img_path)
            
            if diameter is not None:
                results.append([filename, f"{diameter:.2f}"])
                print(f"Width: {diameter:.2f}")
            else:
                results.append([filename, "N/A"])
                print("Failed.")
                
        except Exception as e:
            print(f"Error: {e}")
            results.append([filename, "Error"])

    # Save to CSV
    csv_path = os.path.join(folder_path, "cortex_width_results.csv")
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "Cortex Width (Pixels)"])
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