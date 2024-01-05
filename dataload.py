import librosa
import numpy as np
import pathlib
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import random

class AudioDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Augmentation functions
def add_noise(data, noise_level=0.005):
    noise = np.random.randn(len(data))
    return data + noise_level * noise

def time_stretch(data, rate=0.8):
    return librosa.effects.time_stretch(y = data, rate = rate)

def pitch_shift(data, sampling_rate, n_steps=5):
    return librosa.effects.pitch_shift(y=data,sr = sampling_rate, n_steps = n_steps)

def apply_augmentation(data, sr):
    augmentations = [
        lambda x: add_noise(x),
        lambda x: time_stretch(x, rate=random.uniform(0.8, 1.2)),
        lambda x: pitch_shift(x, sr, n_steps=random.randint(-2, 2))
    ]
    aug_choice = random.choice(augmentations)
    return aug_choice(data)


def get_log_mel_spectrogram_from_audio(y, sr, n_fft=2048, hop_length=512, n_mels=128, target_length=128000):
    # Trim or pad the audio signal
    if len(y) > target_length:
        y = y[:target_length]
    elif len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)), 'constant')

    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    log_mel_spectrogram = librosa.amplitude_to_db(mel_spectrogram)
    return log_mel_spectrogram

def safe_normalize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std == 0] = 1
    return (data - mean) / std

def load_data(path, augmentation_count=3):
    wav_path = pathlib.Path(path)
    files = list(wav_path.glob('*.wav'))
    labels = {'siren': 0, 'bark': 1, 'shoot': 2, 'scream': 3, 'glass': 4, 'alarm': 5, 'explosion': 6}

    original_data = []
    augmented_data = []
    original_targets = []
    augmented_targets = []

    for file in files:
        print(file)
        file_name = file.stem.lower()  # Convert filename to lowercase for comparison
        label_found = False

        for word, label in labels.items():
            if word in file_name:
                label_found = True
                break

        if label_found:
            y, sr = librosa.load(file, sr=16000, duration=4)
            spectrogram = get_log_mel_spectrogram_from_audio(y, sr)
            original_data.append(spectrogram)
            original_targets.append(labels[word])
            
            # Data augmentation
            for _ in range(augmentation_count - 1):
                y_augmented = apply_augmentation(y, sr)
                spectrogram_augmented = get_log_mel_spectrogram_from_audio(y_augmented, sr)
                augmented_data.append(spectrogram_augmented)
                augmented_targets.append(labels[word])

    original_data = np.array(original_data)
    augmented_data = np.array(augmented_data)
    original_targets = np.array(original_targets)
    augmented_targets = np.array(augmented_targets)

    # Split original data for testing and validation
    X_train, X_test, y_train, y_test = train_test_split(original_data, original_targets, test_size=0.2, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.25, random_state=42)  # 0.25 x 0.2 = 0.05 of total data

    # Add augmented data to training data
    X_train = np.concatenate((X_train, augmented_data))
    y_train = np.concatenate((y_train, augmented_targets))

    # Normalise the data
    X_train = safe_normalize(X_train)
    X_test = safe_normalize(X_test)
    X_val = safe_normalize(X_val)
    

    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
    X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1], X_val.shape[2])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])

    train_dataset = AudioDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = AudioDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_dataset = AudioDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader