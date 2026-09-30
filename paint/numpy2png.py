import numpy as np
from PIL import Image
import os
import glob

def save_palette_png(numpy_array, file_path, color_table):
    """
    Saves a NumPy array as a palette PNG image.

    Args:
        numpy_array (np.ndarray): A 2D NumPy array with integer values
                                  representing palette indices.
        file_path (str): The path to save the output PNG file.
        color_table (np.ndarray): A NumPy array of shape (N, 3) where N is
                                  the number of colors in the palette.
                                  Each row is an [R, G, B] value.
    """
    if numpy_array.dtype != np.uint8:
        print("Warning: Converting array to np.uint8. Data loss may occur.")
        numpy_array = numpy_array.astype(np.uint8)

    # Create a Pillow image from the NumPy array in palette mode ('P')
    # The values in the array are treated as indices into the color palette.
    image = Image.fromarray(numpy_array, mode='P')

    # The color table needs to be flattened to a 1D list of [R, G, B, R, G, B, ...]
    palette = color_table.flatten().tolist()
    
    # The palette must be padded to 768 values (256 colors * 3 channels) if it's shorter
    if len(palette) < 768:
        palette.extend([0] * (768 - len(palette)))
    
    # Apply the custom color palette to the image
    image.putpalette(palette)

    # Save the image
    image.save(file_path)
    print(f"Image saved to {file_path}")


# --- Example Usage ---

# 1. Define your color table
COLOR_TABLE = np.array([
    [0, 0, 0],       # 0: Background
    [255, 156, 0],   # 1: Cortex
    [0, 0, 255],     # 2: Metaxylem
    [255, 0, 255],   # 3: Stele Tissue
    [255, 0, 0],     # 4: Aerenchyma
    [255, 255, 0],   # 5: Vascular Bundle
    [125, 60, 152],  # 6: Sclerenchyma
    [255, 3, 127],   # 7: Epidermis
    [3, 252, 69]     # 8: Endodermis
], dtype=np.uint8)



def process_directory(directory):
    """
    Finds all .npy files in a directory and converts them to palette PNGs.
    """
    print(f"Scanning for .npy files in '{directory}'...")
    
    # Create a search pattern to find all .npy files in the directory
    search_pattern = os.path.join(directory, '*.npy')
    npy_files = glob.glob(search_pattern)

    if not npy_files:
        print("No .npy files found in the specified directory. Exiting.")
        return

    print(f"Found {len(npy_files)} files. Starting conversion...")
    
    # Loop through each found .npy file
    for npy_path in npy_files:
        try:
            print(f"\nProcessing '{os.path.basename(npy_path)}'...")
            
            # Load the numpy array
            label_array = np.load(npy_path)
            
            # Define the output path by replacing the extension from .npy to .png
            base_filename = os.path.splitext(npy_path)[0]
            output_path = base_filename + '.png'
            
            # Save the array as a palette image
            save_palette_png(label_array, output_path, COLOR_TABLE)
            
            print(f"  ✅ Successfully saved to '{os.path.basename(output_path)}'")

        except Exception as e:
            print(f"  ❌ Failed to process {os.path.basename(npy_path)}. Error: {e}")

    print("\nConversion process finished.")

INPUT_DIRECTORY = '/data/ezajk13/plant/wheat/labels'


# Run the main function
if __name__ == "__main__":
    process_directory(INPUT_DIRECTORY)