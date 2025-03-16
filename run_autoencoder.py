import argparse

# Import your run_inference_on_folder function from inference_autoencoder
from autoencoder.inference_autoencoder import run_inference_on_folder

def parse_args():
    parser = argparse.ArgumentParser(description="Run 1D autoencoder inference on TIF files.")
    parser.add_argument("-i", "--tiff_input_path", type=str, required=True,
                        help="Path to folder containing .tif input files.")
    parser.add_argument("-o", "--output_path", type=str, default="./inference_results",
                        help="Output folder for reconstructed .tif files.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained autoencoder model (.pth).")
    parser.add_argument("--segment_length", type=int, default=100,
                        help="Length of each 1D segment.")
    parser.add_argument("--interval", type=int, default=50,
                        help="Step size for segment extraction.")
    parser.add_argument("--clip_limit", type=float, default=3,
                        help="Clip limit in lnorm.")
    return parser.parse_args()

def main():
    # Parse the command-line arguments
    args = parse_args()

    # Call the function from your inference_autoencoder module
    run_inference_on_folder(
        tiff_input_path=args.tiff_input_path,
        output_path=args.output_path,
        model_path=args.model_path,
        segment_length=args.segment_length,
        interval=args.interval,
        clip_limit=args.clip_limit,
    )

if __name__ == "__main__":
    main()
