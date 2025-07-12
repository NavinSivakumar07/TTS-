import pandas as pd
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from src import config
from src.utils import text_to_sequence, pad_sequence

class TextMelDataset(Dataset):
    """
    A PyTorch Dataset for loading text and mel-spectrogram pairs.
    """
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.text_col = config.TEXT_COL
        self.mel_col = config.MEL_COL

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get text and mel-spectrogram
        text = self.df.iloc[idx][self.text_col]
        mel_json = self.df.iloc[idx][self.mel_col]

        # Convert text to a sequence of character IDs
        text_seq = text_to_sequence(text)

        # Parse mel-spectrogram from JSON string
        mel_spectrogram = json.loads(mel_json)
        
        # Reshape the flattened mel-spectrogram to (n_mels, length)
        # The data is 80x160 = 12800 floats
        mel_reshaped = np.array(mel_spectrogram).reshape(
            config.MEL_CHANNELS, config.MEL_SPEC_LENGTH
        )

        return torch.LongTensor(text_seq), torch.FloatTensor(mel_reshaped)

def get_dataloader(csv_path, batch_size, shuffle=True):
    """
    Creates a DataLoader for the Text-Mel dataset.
    """
    dataset = TextMelDataset(csv_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=pad_sequence, # Use custom collate function for padding
        num_workers=0 # Set to 0 for Windows compatibility in this context
    )
    return dataloader

if __name__ == '__main__':
    # A simple test to verify the dataloader
    dataloader = get_dataloader(config.DATA_PATH, batch_size=4)
    for i, batch in enumerate(dataloader):
        input_padded, target_padded = batch
        print(f"Batch {i+1}:")
        print("Input (padded text sequence):", input_padded.shape)
        print("Target (mel-spectrogram):", target_padded.shape)
        if i == 0:
            break
    print("\nData loader test complete.")
