import torch
import numpy as np
import librosa
import soundfile as sf
from src.model import SimpleTTS
from src.utils import text_to_sequence
from src import config

def generate_mel(text, model_path="simple_tts_model.pth"):
    """
    Generates a mel-spectrogram from a given text using the trained model.
    """
    # Setup
    device = torch.device(config.DEVICE)
    model = SimpleTTS()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Prepare input text
    sequence = torch.LongTensor(text_to_sequence(text)).unsqueeze(0).to(device)

    # Generate mel-spectrogram
    with torch.no_grad():
        mel_output = model(sequence)

    # Move to CPU and convert to numpy
    mel_output_np = mel_output.squeeze(0).cpu().numpy()

    return mel_output_np

def save_wav(mel_spectrogram, path):
    """Converts a mel-spectrogram to a WAV file using Griffin-Lim."""
    # This is a simplified inversion using Griffin-Lim.
    stft_matrix = librosa.feature.inverse.mel_to_stft(
        mel_spectrogram,
        sr=config.SAMPLING_RATE,
        n_fft=config.N_FFT
    )
    audio = librosa.griffinlim(stft_matrix, hop_length=config.HOP_LENGTH)
    sf.write(path, audio, config.SAMPLING_RATE)

if __name__ == '__main__':
    # Example usage
    sample_text = "hello world this is a test"
    print(f"Generating mel-spectrogram for: '{sample_text}'")

    # Generate and save the mel-spectrogram
    generated_mel = generate_mel(sample_text)
    mel_output_path = "generated_mel.npy"
    np.save(mel_output_path, generated_mel)
    
    print(f"Mel-spectrogram saved to {mel_output_path}")
    print("Shape of the generated mel:", generated_mel.shape)

    # Convert mel-spectrogram to audio
    print("\nConverting mel-spectrogram to audio using Griffin-Lim...")
    audio_output_path = "generated_audio.wav"
    save_wav(generated_mel, audio_output_path)
    print(f"Audio saved to {audio_output_path}")
