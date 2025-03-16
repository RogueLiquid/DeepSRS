import argparse
import os
import glob

import numpy as np
import tifffile as tiff
from PIL import Image

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Convert TIF depth slices to RGB PNG images.")
    parser.add_argument(
        "--input_folder",
        type=str,
        default=".",
        help="Folder containing input TIF files."
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="png",
        help="Output folder for the PNG images."
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Starting depth index (0-based)."
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=0,
        help="Ending depth index (exclusive). If 0, will use the entire range."
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="If set, search for .tif files recursively."
    )
    return parser.parse_args()


def process_tifs_to_rgb_png(input_folder, output_folder="png", start_index=0, end_index=0, recursive=False):
    """
    Reads .tif files under `input_folder`, selects the appropriate depth slices,
    and saves them as RGB PNG images.

    Args:
        input_folder (str): Path to the folder containing .tif files.
        output_folder (str): The folder where PNG files will be saved. Defaults to "png".
        start_index (int): The starting depth index to process (0-based). Default is 0.
        end_index (int): The ending depth index to process (exclusive). Default is 0, meaning use the entire range.
        recursive (bool): Whether to search for .tif files in subdirectories recursively.
    """
    # Build the glob pattern
    if recursive:
        # Search all subdirectories
        pattern = os.path.join(input_folder, "**", "*.tif")
    else:
        # Search only in the top-level folder
        pattern = os.path.join(input_folder, "*.tif")

    # Find all .tif files
    filenames = glob.glob(pattern, recursive=recursive)
    if not filenames:
        print(f"No .tif files found in '{input_folder}' (recursive={recursive}).")
        return

    for file_path in filenames:
        # Read the TIF data
        image_data = tiff.imread(file_path)

        # If both start and end indices are 0, use the entire range of depths
        if start_index == 0 and end_index == 0:
            slice_start = 0
            slice_end = len(image_data)
        else:
            slice_start = start_index
            slice_end = end_index

        # Select the slices
        selected_slices = image_data[slice_start:slice_end]

        # Compute the relative path (for storing output in a parallel structure)
        # Then replace spaces with underscores
        relative_path = os.path.relpath(file_path, os.path.commonpath(filenames))
        relative_path = relative_path.replace(" ", "_")

        # Build an output subfolder corresponding to this file's directory structure
        output_subfolder = os.path.join(output_folder, os.path.dirname(relative_path))
        os.makedirs(output_subfolder, exist_ok=True)

        for i, slice_data in enumerate(selected_slices):
            # Normalize the slice to [0, 255]
            min_val = np.min(slice_data)
            max_val = np.max(slice_data)
            normalized_slice = (
                (slice_data - min_val) / (max_val - min_val + 1e-8) * 255
            ).astype(np.uint8)

            # Convert single-channel to 3-channel (RGB)
            rgb_slice = np.stack([normalized_slice] * 3, axis=-1)
            img = Image.fromarray(rgb_slice, mode="RGB")

            # e.g., file_name.tif -> file_name_depth0.png
            base_name = os.path.basename(file_path).replace(".tif", "").replace(" ", "_")
            output_filename = os.path.join(
                output_subfolder,
                f"{base_name}_depth{slice_start + i}.png"
            )
            img.save(output_filename)
            print(f"Saved {output_filename}")


def main():
    """
    Main function that parses command-line arguments and processes TIF files to PNG.
    """
    args = parse_args()
    process_tifs_to_rgb_png(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        start_index=args.start_index,
        end_index=args.end_index,
        recursive=args.recursive
    )

if __name__ == "__main__":
    main()
