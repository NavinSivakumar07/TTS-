import torch

# Data parameters
DATA_PATH = "data/toy_tts_parallel_data.csv"
TEXT_COL = "normalized_text"
MEL_COL = "mel_spectrogram"

# Text processing
VOCAB = " abcdefghijklmnopqrstuvwxyz'" # All lowercase + space + '
PAD_TOKEN = "P"
VOCAB_WITH_PAD = VOCAB + PAD_TOKEN

# Model parameters
N_MELS = 80
N_SYMBOLS = len(VOCAB_WITH_PAD)
EMBEDDING_DIM = 256
HIDDEN_DIM = 256
N_ENCODER_LAYERS = 3
N_DECODER_LAYERS = 3
DROPOUT = 0.1

# Training parameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 100 # Set high to demonstrate overfitting
CLIP_THRESH = 1.0

# Mel-spectrogram parameters
# These are fixed based on the dataset description
MEL_SPEC_LENGTH = 160 # 80x160 -> length is 160
MEL_CHANNELS = 80

# Audio parameters for Griffin-Lim
SAMPLING_RATE = 24000
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
