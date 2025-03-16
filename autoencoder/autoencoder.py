import os
import glob
import random
import shutil

import tqdm
import numpy as np
import matplotlib.pyplot as plt
import tifffile

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


##########################
# 1. Helper functions
##########################

def read_tiff(path):
    """
    Reads a single .tif file and returns its contents as a NumPy array.
    """
    return tifffile.imread(path)


def znorm(data):
    """
    Z-normalize the data (subtract mean, divide by std).
    """
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std


def lnorm(data, clip_limit=2):
    """
    Clip the data, then linearly scale to [-1,1].
    """
    data_clipped = np.clip(data, -clip_limit, clip_limit)
    min_val = np.min(data_clipped)
    max_val = np.max(data_clipped)
    return ((data_clipped - min_val) / (max_val - min_val)) * 2 - 1


def generate_segments(origin, segment_length, step):
    """
    Splits the 3D data (Frames, Height, Width) along the 'Frames' dimension
    into segments of length 'segment_length', stepping by 'step'.
    Each segment is reshaped from (segment_length, H, W) to (H*W, segment_length).
    If the last segment is shorter, it is zero-padded.
    """
    segments = np.empty(shape=(0, segment_length))
    start = 0
    while start < origin.shape[0]:
        end = start + segment_length
        segment = origin[start:end, :, :]

        # If not enough frames remain, pad
        if segment.shape[0] < segment_length:
            padding = segment_length - segment.shape[0]
            segment = np.pad(segment, ((0, padding), (0, 0), (0, 0)), mode='constant')
        
        # Reshape from (seg_length, H, W) to (H*W, seg_length)
        segment = segment.transpose(1, 2, 0).reshape(-1, segment_length)
        segments = np.concatenate((segments, segment), axis=0)

        start += step

    return segments


def read_all_tifs(folder_path, segment_length=100, interval=100, clip_limit=2):
    """
    Reads all .tif files in 'folder_path', extracts segments,
    applies z-normalization and lnorm, and concatenates them.
    """
    filenames = glob.glob(os.path.join(folder_path, '*.tif'))
    print(f"Total file count is: {len(filenames)}")
    print("Processing files...")

    out = np.empty(shape=(0, segment_length))
    for name in tqdm.tqdm(filenames):
        tif_data = read_tiff(name)
        segments = generate_segments(tif_data, segment_length, interval)
        segments = znorm(segments)
        segments = lnorm(segments, clip_limit)
        out = np.concatenate((out, segments), axis=0)
    return out


def numpy2tensor(array_data, idxs):
    """
    Takes a NumPy array of shape (N, segment_length), selects rows by idxs,
    and returns a FloatTensor of shape (N_subset, 1, segment_length).
    """
    subsetdata = array_data[idxs, :]
    return torch.tensor(subsetdata, dtype=torch.float32).unsqueeze(1)


##########################
# 2. PyTorch Dataset
##########################

class PixelDataset(Dataset):
    """
    A simple dataset that holds a tensor of shape (N, 1, segment_length).
    """
    def __init__(self, tensor_data):
        self.data = tensor_data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]


##########################
# 3. Model definition
##########################

class Conv1dAutoencoder(nn.Module):
    """
    A 1D convolutional autoencoder.
    Encoder input: (batch_size, 1, 100)
    Decoder output: (batch_size, 1, 100)
    """
    def __init__(self):
        super(Conv1dAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=7, stride=3, padding=0),   # -> (B,4,32)
            nn.Tanh(),
            nn.MaxPool1d(2),                                       # -> (B,4,16)

            nn.Conv1d(4, 8, kernel_size=5, stride=1, padding=0),   # -> (B,8,12)
            nn.Tanh(),
            nn.MaxPool1d(2),                                       # -> (B,8,6)

            nn.Conv1d(8, 16, kernel_size=3, stride=1, padding=0),  # -> (B,16,4)
            nn.Tanh(),

            nn.Conv1d(16, 24, kernel_size=3, stride=1, padding=0), # -> (B,24,2)
            nn.Tanh()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(24, 16, kernel_size=3, stride=1, padding=0),  # -> (B,16,4)
            nn.Tanh(),
            nn.ConvTranspose1d(16, 8, kernel_size=5, stride=1, padding=0),   # -> (B,8,8)
            nn.Tanh(),
            nn.ConvTranspose1d(8, 6, kernel_size=6, stride=2, padding=0),    # -> (B,6,20)
            nn.Tanh(),
            nn.ConvTranspose1d(6, 4, kernel_size=8, stride=2, padding=0),    # -> (B,4,46)
            nn.Tanh(),
            nn.ConvTranspose1d(4, 1, kernel_size=10, stride=2, padding=0),   # -> (B,1,100)
            nn.Tanh()
        )

        # Fully connected layers for the latent space
        self.fc_e = nn.Linear(48, 32)
        self.fc_d = nn.Linear(32, 48)

    def forward(self, x):
        encoded = self.encoder(x)            # -> shape: (B,24,2)
        middle = encoded.view(-1, 48)        # flatten 24*2=48
        middle = self.fc_e(middle)           # -> shape: (B,32)
        middle = self.fc_d(middle)           # -> shape: (B,48)
        middle = middle.view(-1, 24, 2)      # reshape back
        decoded = self.decoder(middle)       # -> shape: (B,1,100)
        return decoded


##########################
# 4. Main training function
##########################

def run_autoencoder_training(
    data_input_path,
    output_path,
    segment_length=100,
    interval=100,
    clip_limit=2,
    train_split=0.7,
    batch_size=256,
    learning_rate=1e-3,
    epochs=50,
    plot_example_flag=False,
    seed=42
):
    """
    Runs the entire 1D autoencoder training pipeline:
      - Reads and normalizes .tif files from data_input_path.
      - Splits data into train/test sets.
      - Defines and trains Conv1dAutoencoder.
      - Saves the best model checkpoints in output_path.

    Args:
        data_input_path (str): Path to the folder containing .tif files.
        output_path (str): Path to the folder where models and logs are saved.
        segment_length (int): Length of each segment extracted from the .tif.
        interval (int): Step size for sliding-window across frames.
        clip_limit (float): Used in lnorm to clamp data before scaling.
        train_split (float): Fraction of data used for training (0 < train_split < 1).
        batch_size (int): Batch size for training and testing.
        learning_rate (float): Learning rate for optimizer.
        epochs (int): Number of training epochs.
        plot_example_flag (bool): Whether to plot and save sample input/output each epoch.
        seed (int): Random seed for reproducibility.

    Returns:
        model (nn.Module): Trained Conv1dAutoencoder model.
    """

    # Ensure reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Renew or create output folder
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
        os.makedirs(output_path)
        print(f"Output folder '{output_path}' is renewed.")
    else:
        os.makedirs(output_path)
        print(f"Output folder '{output_path}' is newly erected.")

    # 1) Read all .tif data
    data = read_all_tifs(data_input_path, segment_length, interval, clip_limit)
    print("the shape of the data is:", data.shape)

    # 2) Split into train/test
    train_num = int(data.shape[0] * train_split)
    test_num = data.shape[0] - train_num

    train_indices = random.sample(range(data.shape[0]), train_num)
    test_indices = list(set(range(data.shape[0])) - set(train_indices))

    train_data = numpy2tensor(data, train_indices)
    test_data = numpy2tensor(data, test_indices)

    print(f"The length of the training dataset is: {train_data.shape[0]}")
    print(f"The length of the testing dataset is: {test_data.shape[0]}")

    # 3) Create dataset and dataloaders
    train_dataset = PixelDataset(train_data)
    test_dataset = PixelDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 4) Setup model, optimizer, loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Conv1dAutoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    print("Model is running on:", device)

    # For saving the best model
    best_test_loss = float('inf')
    best_model_path = 'best_autoencoder_model.pth'
    best_save_count = 0

    # 5) Optional function to plot input vs. output each epoch
    def plot_example_func(input_data, output_data, epoch_num):
        """
        Plots the first example in a batch: input and reconstructed output.
        """
        plt.figure(figsize=(12, 6))
        input_example = input_data[0].squeeze()   # shape: (segment_length,)
        output_example = output_data[0].squeeze() # shape: (segment_length,)

        # Plot input
        plt.subplot(1, 2, 1)
        plt.plot(input_example.cpu().numpy(), label='Input')
        plt.title('Input Data')
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.legend()

        # Plot output
        plt.subplot(1, 2, 2)
        plt.plot(output_example.cpu().numpy(), label='Output')
        plt.title('Output Data')
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.legend()

        plt.savefig(f'example_epoch_{epoch_num + 1}.png')
        plt.close()

    ############################
    # 6) Training loop
    ############################
    for epoch in range(epochs):
        # ---- Train ----
        model.train()
        running_loss = 0.0
        for batch_data in train_loader:
            batch_data = batch_data.to(device)
            output = model(batch_data)
            loss = criterion(output, batch_data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_data.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # ---- Evaluate on test set ----
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for batch_data in test_loader:
                batch_data = batch_data.to(device)
                output = model(batch_data)
                loss = criterion(output, batch_data)
                running_loss += loss.item() * batch_data.size(0)

        test_loss = running_loss / len(test_loader.dataset)

        # ---- Record loss ----
        with open('loss_record.txt', 'a') as f:
            f.write(f'{epoch+1}\t{train_loss:.4f}\t{test_loss:.4f}\n')

        # ---- Check for best model ----
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_save_count += 1

            # Save model every time test_loss improves
            # but keep a "tagged" version every 4 saves
            if best_save_count % 4 == 0:
                base_name = best_model_path.replace(".pth", "__")
                model_name = base_name + f'{best_save_count}.pth'
                output_model_path = os.path.join(output_path, model_name)
                torch.save(model.state_dict(), output_model_path)
                print(f'New best model saved at epoch {epoch+1} with test loss: {best_test_loss:.4f}')

        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

        # ---- Optional: plot sample input vs. output
        if plot_example_flag:
            # Use last batch from test loop for plotting
            plot_example_func(batch_data, output, epoch)

    print("Training completed.")

    # Return the trained model if further usage is desired
    return model