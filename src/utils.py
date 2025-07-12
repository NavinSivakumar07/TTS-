import torch
from src import config

# Create character-to-index and index-to-character mappings
char_to_idx = {char: i for i, char in enumerate(config.VOCAB_WITH_PAD)}
idx_to_char = {i: char for i, char in enumerate(config.VOCAB_WITH_PAD)}
PAD_IDX = char_to_idx[config.PAD_TOKEN]

def text_to_sequence(text):
    """Converts a string of text to a sequence of IDs."""
    sequence = [
        char_to_idx.get(char, -1)
        for char in text.lower()
        if char in char_to_idx
    ]
    return sequence

def sequence_to_text(sequence):
    """Converts a sequence of IDs back to a string."""
    return "".join([idx_to_char.get(idx, "") for idx in sequence])

def pad_sequence(batch):
    """
    Pads sequences to the maximum length in a batch.
    This custom collate function is used by the DataLoader.
    """
    # Separate inputs and targets
    input_seqs, target_mels = zip(*batch)

    # Pad input sequences
    max_len_in = max(len(s) for s in input_seqs)
    input_padded = torch.LongTensor(len(batch), max_len_in).fill_(PAD_IDX)
    for i, seq in enumerate(input_seqs):
        input_padded[i, :len(seq)] = torch.LongTensor(seq)

    # Pad target mel-spectrograms
    # The mel spectrograms in the toy dataset are of fixed size,
    # but in a real-world scenario, they would also need padding.
    # Here, we just stack them.
    target_padded = torch.stack(target_mels, 0)

    return input_padded, target_padded
