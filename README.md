# Minimal TTS Engine for India Speaks

This project is a proof-of-concept Text-to-Speech (TTS) engine for generating Tamil speech, tailored for low-power mobile devices. It's built with PyTorch and designed to be lightweight and fast.

## Project Overview

The goal of this project is to create a minimal, well-structured TTS system that can be trained on a small parallel dataset of text and mel-spectrograms. The architecture is a simplified FastSpeech-style model, consisting of an encoder and a decoder, designed for real-time inference.

This implementation successfully demonstrates that the model can overfit the provided toy dataset, verifying the correctness of the end-to-end pipeline.

### Key Features

*   **Lightweight Architecture:** A simple encoder-decoder model suitable for mobile deployment.
*   **PyTorch Implementation:** Built with modern and flexible tools.
*   **Clear Structure:** Organized into `src` for source code and `data` for datasets.
*   **End-to-End Pipeline:** Includes data loading, training, and inference scripts.

## Project Structure

```
TTS/
├── data/
│   └── toy_tts_parallel_data.csv
├── src/
│   ├── __init__.py
│   ├── config.py           # All hyperparameters and settings
│   ├── data_loader.py      # PyTorch Dataset and DataLoader
│   ├── model.py            # The TTS model architecture
│   ├── train.py            # The main training script
│   ├── inference.py        # Script to generate audio from text
│   └── utils.py            # Text processing utilities
├── requirements.txt        # Project dependencies
├── simple_tts_model.pth    # Trained model weights
├── generated_mel.npy       # A sample generated mel-spectrogram
└── README.md               # This file
```

## Getting Started

### Prerequisites

*   Python (>=3.8)
*   PyTorch
*   Pandas, NumPy, Librosa

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd TTS
    ```

2.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

### How to Run

1.  **Train the model:**
    To train the model from scratch, run the training script. This will train the model on the toy dataset and save the weights as `simple_tts_model.pth`.
    ```bash
    python -m src.train
    ```
    The training is set to run for 100 epochs to demonstrate overfitting. You can adjust this and other parameters in `src/config.py`.

2.  **Generate a mel-spectrogram (Inference):**
    To generate a mel-spectrogram from a sample text using the trained model, run the inference script.
    ```bash
    python -m src.inference
    ```
    This will use the `simple_tts_model.pth` file to generate `generated_mel.npy`. You can change the input text inside the `src/inference.py` script.
