import os
import torch

from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np

import logging

# Set up logging.
logging.basicConfig(level=logging.INFO)


class SpecAugmentationTransform:
    def __init__(self, augment=True):
        """
        Args:
            augment (bool): If True, perform data augmentation.
        """
        self.augment = augment

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation and normalization on the spectrogram tensor.

        Args:
            spec (torch.Tensor): Spectrogram tensor of shape [freq, time].

        Returns:
            torch.Tensor: Transformed spectrogram with shape [1, freq, time].
        """
        # Apply augmentation if enabled.
        if self.augment:
            spec = self.apply_specaugment(spec)

        # Normalize per spectrogram.
        mean = spec.mean()
        std = spec.std()
        spec = (spec - mean) / (std + 1e-6)

        # Add channel dimension if not present.
        if spec.ndim == 2:
            spec = spec.unsqueeze(0)
        return spec

    def apply_specaugment(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Applies simple SpecAugment: random frequency and time masking.

        Args:
            spec (torch.Tensor): Spectrogram tensor of shape [freq, time].

        Returns:
            torch.Tensor: Augmented spectrogram.
        """
        freq_bins, time_steps = spec.shape

        # Frequency masking: mask up to 10% of the frequency bins.
        freq_mask_param = int(0.1 * freq_bins)
        if freq_mask_param > 0:
            freq_mask_width = np.random.randint(0, freq_mask_param)
            if freq_mask_width > 0:
                freq_start = np.random.randint(0, max(1, freq_bins - freq_mask_width))
                spec[freq_start:freq_start + freq_mask_width, :] = 0.0

        # Time masking: mask up to 10% of the time steps.
        time_mask_param = int(0.1 * time_steps)
        if time_mask_param > 0:
            time_mask_width = np.random.randint(0, time_mask_param)
            if time_mask_width > 0:
                time_start = np.random.randint(0, max(1, time_steps - time_mask_width))
                spec[:, time_start:time_start + time_mask_width] = 0.0

        return spec


class SpectrogramDataset(Dataset):
    def __init__(self, spectrograms, labels, transform=None):
        self.transform = transform
        self.spectrograms = [torch.tensor(spec, dtype=torch.float32) for spec in spectrograms]
        unique_labels = sorted(set(labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.labels = [self.label_to_idx[label] for label in labels]

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, idx):
        sample = self.spectrograms[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label


def extract_spectrograms(
        audio_path: str,
        window_duration: float,
        sr: int = 22050,
        n_fft: int = 2048,
        hop_length: int = None
) -> list:
    if hop_length is None:
        hop_length = n_fft // 4

    try:
        audio, sr = librosa.load(audio_path, sr=sr)
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return []

    window_length = int(window_duration * sr)
    step_size = int(window_length * 0.8)  # 20% overlap
    spectrograms = []
    for start in range(0, len(audio) - window_length + 1, step_size):
        window = audio[start:start + window_length]
        S = librosa.stft(window, n_fft=n_fft, hop_length=hop_length)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        spectrograms.append(S_db)

    return spectrograms


def create_dataset(audio_dir: str, window_duration: float, sr: int = 22050,
                   n_fft: int = 2048, hop_length: int = None):
    X = []
    y = []
    for file_name in os.listdir(audio_dir):
        if file_name.lower().endswith(('.wav', '.mp3', '.flac')):
            file_path = os.path.join(audio_dir, file_name)
            specs = extract_spectrograms(file_path, window_duration, sr, n_fft, hop_length)
            label = os.path.splitext(file_name)[0]
            for spec in specs:
                X.append(spec)
                y.append(label)
    return X, y


def create_dataloader(spectrograms, labels, batch_size=32, shuffle=True, transform=None):
    dataset = SpectrogramDataset(spectrograms, labels, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
