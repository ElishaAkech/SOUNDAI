import os
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import soundfile as sf
import librosa

# Define class labels
class_mapping = [
    "bicycle", "Motorcycle", "car", "pickup", "SUV", "PSV", "Buses",
    "Light Trucks", "Medium trucks", "Heavy trucks", "car_horn", "drilling",
    "engine_idling", "siren"
]

# Paths
wav_folder = "data\\DATA\\output_chunks"  # Replace with your WAV folder path
csv_path = "data\\DATA\\output_chunks\\metadata.csv"         # Replace with your metadata CSV path
dataset_path = "data\\DATA\\output_chunks"  # Replace with labeled dataset path (for fine-tuning)
model_save_path = "finetuned_yamnet"  # Path to save/load fine-tuned model

# Step 1: Load and preprocess audio
def load_wav_16k_mono(file_path):
    """Load WAV file, resample to 16 kHz mono."""
    try:
        y, sr = sf.read(file_path)
        if len(y.shape) > 1:  # Convert stereo to mono
            y = np.mean(y, axis=1)
        if sr != 16000:  # Resample to 16 kHz
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        return y
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Step 2: Prepare dataset for fine-tuning (optional)
def prepare_dataset(dataset_path):
    """Load WAV files and labels from dataset folder."""
    wavs, labels = [], []
    for idx, class_label in enumerate(class_mapping):
        class_folder = os.path.join(dataset_path, class_label)
        if not os.path.exists(class_folder):
            continue
        for file_name in os.listdir(class_folder):
            if file_name.endswith(".wav"):
                file_path = os.path.join(class_folder, file_name)
                wav = load_wav_16k_mono(file_path)
                if wav is not None:
                    wavs.append(wav)
                    labels.append(idx)
    return np.array(wavs, dtype=object), np.array(labels)

# Step 3: Build fine-tuned model
def build_finetuned_model(num_classes):
    """Load YAMNet and add a new classification layer."""
    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
    yamnet_model = yamnet_model.signatures['serving_default']
    inputs = tf.keras.Input(shape=(None,), dtype=tf.float32, name='audio_input')
    _, embeddings, _ = yamnet_model(inputs)
    x = tf.keras.layers.Dense(512, activation='relu')(embeddings)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Step 4: Fine-tune model (optional)
def train_model(dataset_path, model_save_path):
    """Fine-tune YAMNet on your dataset."""
    wavs, labels = prepare_dataset(dataset_path)
    if len(wavs) == 0:
        raise ValueError("No valid WAV files found in dataset.")
    
    from sklearn.model_selection import train_test_split
    wavs_train, wavs_test, labels_train, labels_test = train_test_split(
        wavs, labels, test_size=0.2, random_state=42
    )

    model = build_finetuned_model(len(class_mapping))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    def preprocess_wavs(wavs, max_length=16000 * 10):
        processed = []
        for wav in wavs:
            if len(wav) > max_length:
                wav = wav[:max_length]
            else:
                wav = np.pad(wav, (0, max_length - len(wav)), mode='constant')
            processed.append(wav)
        return np.array(processed)

    wavs_train = preprocess_wavs(wavs_train)
    wavs_test = preprocess_wavs(wavs_test)

    model.fit(
        wavs_train, labels_train,
        validation_data=(wavs_test, labels_test),
        epochs=10, batch_size=32, verbose=1
    )

    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    return model

# Step 5: Map pre-trained YAMNet classes (fallback if not fine-tuning)
def map_yamnet_to_custom(predicted_class, yamnet_classes):
    """Map YAMNet's 521 classes to your 14 classes."""
    # Load YAMNet class labels (available from TensorFlow Hub)
    yamnet_class_map = pd.read_csv(
        'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
    )
    yamnet_class_name = yamnet_classes[predicted_class]
    # Simple mapping (customize based on your needs)
    mapping_dict = {
        'Car horn': 'car_horn',
        'Siren': 'siren',
        'Engine idling': 'engine_idling',
        'Drill': 'drilling',
        'Motorcycle': 'Motorcycle',
        'Car': 'car',  # May map to car, SUV, pickup, etc.
        'Bus': 'Buses',
        'Truck': 'Heavy trucks'  # May map to Light/Medium/Heavy trucks
        # Add more mappings as needed
    }
    return mapping_dict.get(yamnet_class_name, 'unknown')  # Default to 'unknown'

# Step 6: Classify WAV files
def classify_wav_files(wav_folder, model=None, use_pretrained=False):
    """Classify WAV files using fine-tuned or pre-trained YAMNet."""
    labels = []
    if use_pretrained:
        yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
        yamnet_model = yamnet_model.signatures['serving_default']
        # Load YAMNet class labels
        yamnet_class_map = pd.read_csv(
            'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
        )
        yamnet_classes = yamnet_class_map['display_name'].values

    for file_name in os.listdir(wav_folder):
        if file_name.endswith(".wav"):
            file_path = os.path.join(wav_folder, file_name)
            wav = load_wav_16k_mono(file_path)
            if wav is None:
                continue
            # Pad or truncate to 10 seconds
            max_length = 16000 * 10
            if len(wav) > max_length:
                wav = wav[:max_length]
            else:
                wav = np.pad(wav, (0, max_length - len(wav)), mode='constant')
            wav = wav[np.newaxis, :]  # Add batch dimension

            if use_pretrained:
                # Use pre-trained YAMNet
                scores, _, _ = yamnet_model(wav)
                predicted_class = np.argmax(scores, axis=1)[0]
                predicted_label = map_yamnet_to_custom(predicted_class, yamnet_classes)
            else:
                # Use fine-tuned model
                prediction = model.predict(wav, verbose=0)
                predicted_label = class_mapping[np.argmax(prediction, axis=1)[0]]

            labels.append({"filename": file_name, "class_label": predicted_label})
    return labels

# Step 7: Update metadata CSV
def update_metadata_csv(csv_path, labels):
    """Update metadata CSV with predicted labels."""
    try:
        metadata = pd.read_csv(csv_path)
        new_data = pd.DataFrame(labels)
        if "class_label" in metadata.columns:
            metadata = metadata.drop(columns=["class_label"])
        metadata = metadata.merge(new_data, on="filename", how="left")
        metadata.to_csv("updated_metadata.csv", index=False)
        print("Metadata CSV updated successfully.")
    except Exception as e:
        print(f"Error updating CSV: {e}")

# Main execution
if __name__ == "__main__":
    # Decide whether to use fine-tuned or pre-trained model
    use_pretrained = False  # Set to True to use pre-trained YAMNet

    if not use_pretrained:
        # Load or train fine-tuned model
        if os.path.exists(model_save_path):
            print(f"Loading fine-tuned model from {model_save_path}")
            model = tf.keras.models.load_model(model_save_path)
        else:
            print("No fine-tuned model found. Training new model...")
            model = train_model(dataset_path, model_save_path)
    else:
        model = None  # Not needed for pre-trained YAMNet

    # Classify WAV files
    predicted_labels = classify_wav_files(wav_folder, model, use_pretrained=use_pretrained)

    # Update CSV
    update_metadata_csv(csv_path, predicted_labels)