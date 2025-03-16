import os
import glob
import shutil

import numpy as np
import torch
import torch.nn as nn
import tifffile
import matplotlib.pyplot as plt


##########################
# 1. Model definition
##########################

class Conv1dAutoencoder(nn.Module):
    def __init__(self):
        super(Conv1dAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=7, stride=3, padding=0),  # (batch_size, 4, 32)
            nn.Tanh(),
            nn.MaxPool1d(2),                                      # (batch_size, 4, 16)

            nn.Conv1d(4, 8, kernel_size=5, stride=1, padding=0),  # (batch_size, 8, 12)
            nn.Tanh(),
            nn.MaxPool1d(2),                                      # (batch_size, 8, 6)

            nn.Conv1d(8, 16, kernel_size=3, stride=1, padding=0), # (batch_size, 16, 4)
            nn.Tanh(),

            nn.Conv1d(16, 24, kernel_size=3, stride=1, padding=0),# (batch_size, 24, 2)
            nn.Tanh()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(24, 16, kernel_size=3, stride=1, padding=0),  # (batch_size, 16, 4)
            nn.Tanh(),
            nn.ConvTranspose1d(16, 8, kernel_size=5, stride=1, padding=0),   # (batch_size, 8, 8)
            nn.Tanh(),
            nn.ConvTranspose1d(8, 6, kernel_size=6, stride=2, padding=0),    # (batch_size, 6, 20)
            nn.Tanh(),
            nn.ConvTranspose1d(6, 4, kernel_size=8, stride=2, padding=0),    # (batch_size, 4, 46)
            nn.Tanh(),
            nn.ConvTranspose1d(4, 1, kernel_size=10, stride=2, padding=0),   # (batch_size, 1, 100)
            nn.Tanh()
        )

        # Fully connected layers
        self.fc_e = nn.Linear(48, 32)
        self.fc_d = nn.Linear(32, 48)

    def forward(self, x):
        # Encoder
        encoded = self.encoder(x)
        # Flatten and pass through fully-connected layers
        middle = encoded.view(-1, 48)
        middle = self.fc_e(middle)
        middle = self.fc_d(middle)
        middle = middle.view(-1, 24, 2)
        # Decoder
        decoded = self.decoder(middle)
        return decoded


##########################
# 2. Helper functions
##########################

def read_tiff(path):
    """
    Reads a single .tif file and returns its contents as a NumPy array
    plus the original dtype (so we can cast back later).
    """
    file_data = tifffile.imread(path)
    file_dtype = file_data.dtype
    return file_data, file_dtype

def znorm(data):
    """
    Z-normalizes data: (data - mean) / std.
    Returns the normalized data, plus mean and std for later 'undo'.
    """
    mean = np.mean(data)
    std = np.std(data)
    normalized = (data - mean) / (std + 1e-8)
    return normalized, mean, std

def lnorm(data, clip_limit=3):
    """
    Clips data to [-clip_limit, clip_limit] and rescales to [-1, 1].
    Returns the rescaled data, plus old min and old max for later 'undo'.
    """
    data_clipped = np.clip(data, -clip_limit, clip_limit)
    min_val = np.min(data_clipped)
    max_val = np.max(data_clipped)
    scaled = ((data_clipped - min_val) / (max_val - min_val + 1e-8)) * 2.0 - 1.0
    return scaled, min_val, max_val

def generate_segments(origin, segment_length, interval):
    """
    Splits 'origin' (shape: [Frames, H, W]) into 1D segments of size segment_length.
    Steps forward by 'interval' frames each time.
    Reshapes (segment_length, H, W) into (H*W, segment_length).
    Returns a 2D array of shape [N_segments, segment_length].
    """
    segments = np.empty(shape=(0, segment_length), dtype=np.float32)
    start = 0
    while start < origin.shape[0]:
        end = start + segment_length
        segment = origin[start:end, :, :]  # shape: [seg_len, H, W]
        # If not enough frames, pad
        if segment.shape[0] < segment_length:
            padding = segment_length - segment.shape[0]
            segment = np.pad(segment, ((0, padding), (0, 0), (0, 0)), mode='constant')
        # Now shape -> (H, W, seg_len) -> (H*W, seg_len)
        segment_2d = segment.transpose(1, 2, 0).reshape(-1, segment_length)
        segments = np.concatenate((segments, segment_2d), axis=0)
        start += interval
    return segments

def reconstruct_segments(segments, original_shape, segment_length, interval):
    """
    Reverses what 'generate_segments' did:
    Takes 2D array 'segments' of shape [N_segments, segment_length],
    and accumulates them into a 3D array [Frames, H, W] of shape 'original_shape'.

    'original_shape' is the shape of the original data (Frames, H, W).
    """
    reconstructed = np.zeros(original_shape, dtype=np.float32)
    overlap_counter = np.zeros(original_shape, dtype=np.float32)

    H = original_shape[1]
    W = original_shape[2]
    # We know each "segment" came from a chunk of length 'segment_length'
    # and each chunk was flattened to shape (H*W, seg_len).
    # So we can reshape segments back to: (num_chunks, H*W, segment_length).
    # Then reconstruct frame by frame.
    # Figure out how many "chunks" we have:
    # If generate_segments was called step by step, the number of chunks is
    #   total_segments / (H*W).
    total_chunks = segments.shape[0] // (H * W)
    reshaped_segments = segments.reshape(total_chunks, H*W, segment_length)

    start_frame = 0
    for i in range(total_chunks):
        # Reconvert to (segment_length, H, W)
        chunk_3d = reshaped_segments[i].reshape(H, W, segment_length).transpose(2, 0, 1)
        end_frame = start_frame + segment_length
        if end_frame > original_shape[0]:
            end_frame = original_shape[0]
        # Accumulate
        reconstructed[start_frame:end_frame, :, :] += chunk_3d[:end_frame - start_frame, :, :]
        overlap_counter[start_frame:end_frame, :, :] += 1.0
        start_frame += interval

    # Avoid division by zero
    overlap_counter[overlap_counter == 0] = 1
    reconstructed = reconstructed / overlap_counter
    return reconstructed

def run_model_on_segments(segments, model):
    """
    Runs a trained model on all segments. 
    segments: 2D NumPy array of shape [N, seg_length].
    Returns the model output in the same shape, as a NumPy array.
    """
    # Convert to torch Tensor, shape (N,1, seg_length)
    input_tensor = torch.from_numpy(segments).float().unsqueeze(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    
    model.eval()
    with torch.no_grad():
        output_tensor = model(input_tensor)  # shape: (N, 1, seg_length)
    # Move back to CPU, drop the channel dim
    output_numpy = output_tensor.squeeze(1).cpu().numpy()
    return output_numpy


##########################
# 3. Comprehensive function
##########################

def run_inference_on_folder(
    tiff_input_path,
    output_path,
    model_path,
    segment_length=100,
    interval=50,
    clip_limit=3
):
    """
    Reads all .tif files from 'tiff_input_path', uses a trained Conv1dAutoencoder
    to reconstruct each file (frame-wise), and saves the reconstructed .tif files
    into 'output_path'. 

    Args:
        tiff_input_path (str): Path to folder containing .tif files.
        output_path (str): Folder where output .tif files are saved.
        model_path (str): Path to the trained model weights (.pth).
        segment_length (int): Length of each 1D segment.
        interval (int): Step size for segment extraction.
        clip_limit (float): For lnorm, data is clipped to [-clip_limit, clip_limit].
    """
    # 1) Create or refresh output_path
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
        os.makedirs(output_path)
        print(f"Inference output folder '{output_path}' is renewed.")
    else:
        os.makedirs(output_path)
        print(f"Inference output folder '{output_path}' is created.")

    # 2) Instantiate model and load weights
    model = Conv1dAutoencoder()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 3) Process all .tif files in the folder
    filenames = glob.glob(os.path.join(tiff_input_path, '*.tif'))
    print(f"Total file count is: {len(filenames)}")

    for idx, file_path in enumerate(filenames, 1):
        print(f"Processing file {idx} of {len(filenames)}: {file_path}")
        # Read data
        original_data, original_dtype = read_tiff(file_path)
        original_shape = original_data.shape  # (frames, H, W)

        # 4) Convert data into segments
        segments = generate_segments(original_data, segment_length, interval)

        # 5) z-norm
        segments, z_mean, z_std = znorm(segments)
        # 6) l-norm
        segments, l_min, l_max = lnorm(segments, clip_limit=clip_limit)

        # 7) Run the model
        output_segments = run_model_on_segments(segments, model)

        # 8) Reconstruct from segments
        reconstructed_data = reconstruct_segments(output_segments, original_shape, segment_length, interval)

        # 9) Invert lnorm
        reconstructed_data = ((reconstructed_data + 1.0) / 2.0) * (l_max - l_min) + l_min

        # 10) Invert znorm
        reconstructed_data = reconstructed_data * (z_std + 1e-8) + z_mean

        # 11) Cast back to original dtype
        reconstructed_data = reconstructed_data.astype(original_dtype)

        # 12) Save to disk
        basename = os.path.basename(file_path)
        output_name = basename.replace('.tif', '_Reconstructed.tif')
        out_path = os.path.join(output_path, output_name)
        print(f"Saving reconstructed file as: {output_name}")
        tifffile.imwrite(out_path, reconstructed_data)

    print("Inference completed for all files.")
