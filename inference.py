import torch
import torchaudio
from cnn import CNNNetwork
import os

# Constants from the original code
SAMPLE_RATE = 22050  # Assuming this from UrbanSoundDataset, adjust if different
NUM_SAMPLES = 22050  # Assuming 1 second at 22050 Hz, adjust if different

class_mapping = [
    "bicycle",
    "Motorcycle",
    "car",
    "pickup",
    "SUV",
    "PSV",
    "Buses",
    "Light Trucks",
    "Medium trucks",
    "Heavy trucks",
    "car_horn",    
    "drilling",
    "engine_idling",
    "siren",
]
# End of replacement section

def preprocess_audio(audio_path, sample_rate=SAMPLE_RATE, num_samples=NUM_SAMPLES):
    """
    Load and preprocess an audio file to match the model's input requirements.
    
    Args:
        audio_path (str): Path to the audio file (e.g., WAV)
        sample_rate (int): Target sample rate
        num_samples (int): Number of samples to pad or truncate to
    
    Returns:
        torch.Tensor: Preprocessed audio tensor [1, num_channels, time]
    """
    # Load audio file
    waveform, sr = torchaudio.load(audio_path)
    
    # Resample if necessary
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Pad or truncate to num_samples
    if waveform.shape[1] > num_samples:
        waveform = waveform[:, :num_samples]
    else:
        pad_size = num_samples - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_size))
    
    # Convert to mel spectrogram (mimicking UrbanSoundDataset)
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
    mel_spec = mel_spectrogram(waveform)
    
    # Add channel dimension for model input [batch, channels, freq, time]
    mel_spec = mel_spec.unsqueeze(0)
    
    return mel_spec

def predict(model, input_tensor, class_mapping):
    """
    Predict the class of the input audio.
    
    Args:
        model: Trained CNNNetwork
        input_tensor: Preprocessed audio tensor
        class_mapping: List of class names
    
    Returns:
        tuple: (predicted_class, expected_class) - expected is None for external files
    """
    model.eval()
    with torch.no_grad():
        predictions = model(input_tensor)
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
    return predicted, None  # No expected class for external files

if __name__ == "__main__":
    # Load the pre-trained model with map_location='cpu' to handle CPU-only environments
    cnn = CNNNetwork()
    state_dict = torch.load("cnnnet.pth", map_location=torch.device('cpu'))
    cnn.load_state_dict(state_dict)
    cnn.eval()

    # Prompt user for audio file path
    audio_path = input("Enter the path to the audio file (e.g., WAV from data/newwav): ").strip()
    
    # Validate audio file
    if not os.path.exists(audio_path):
        print(f"Error: File {audio_path} does not exist.")
    elif not audio_path.lower().endswith(('.wav', '.mp3', '.flac')):  # Adjust supported formats as needed
        print(f"Error: {audio_path} is not a supported audio file (WAV, MP3, FLAC).")
    else:
        try:
            # Preprocess the audio file
            input_tensor = preprocess_audio(audio_path)
            
            # Make prediction
            predicted, expected = predict(cnn, input_tensor, class_mapping)
            print(f"Predicted: '{predicted}', expected: '{expected}'")
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
