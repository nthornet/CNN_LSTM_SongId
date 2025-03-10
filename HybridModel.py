import torch
import torch.nn as nn
import torch.optim as optim
import logging
import FileLoading as FL

# Set up logging.
logging.basicConfig(level=logging.INFO)


class CNNLSTM(nn.Module):
    def __init__(self, num_classes: int, lstm_hidden_size: int = 64,
                 lstm_layers: int = 1, dropout: float = 0.5):
        super(CNNLSTM, self).__init__()

        # CNN encoder to extract spatial features from spectrograms.
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.Conv2d(128, 256, kernel_size=3, padding=1),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        # Determine LSTM input size dynamically using a dummy input.
        dummy_input = torch.zeros(1, 1, 1025, 87)
        with torch.no_grad():
            cnn_out = self.cnn(dummy_input)
        _, cnn_channels, cnn_freq, cnn_time = cnn_out.size()
        lstm_input_size = cnn_channels * cnn_freq

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        cnn_features = self.cnn(x)  # (batch, channels, new_freq, new_time)
        cnn_features = cnn_features.permute(0, 3, 1, 2)  # (batch, new_time, channels, new_freq)
        batch_size, seq_len, channels, freq = cnn_features.size()
        cnn_features = cnn_features.reshape(batch_size, seq_len, channels * freq)
        lstm_out, _ = self.lstm(cnn_features)
        last_output = lstm_out[:, -1, :]
        out = self.fc(last_output)
        return out


def classify_new_audio(
        model,
        audio_path: str,
        window_duration: float,
        sr: int = 22050,
        n_fft: int = 2048,
        hop_length: int = None,
        transform=None,
        idx_to_label: dict = None
):
    """
    Classify a new audio file using the trained CNN-LSTM model.

    This function extracts spectrograms from the input audio file using a sliding window with
    20% overlap, then applies the model to each window. It aggregates predictions via a majority vote.

    Args:
        model (nn.Module): Trained CNN-LSTM model.
        audio_path (str): Path to the new audio file.
        window_duration (float): Duration (in seconds) for each spectrogram window.
        sr (int): Sampling rate.
        n_fft (int): Number of FFT components.
        hop_length (int): Hop length for STFT (if None, defaults to n_fft//4).
        transform (callable, optional): Transform to apply to each spectrogram (e.g., add a channel dimension).
        idx_to_label (dict, optional): Mapping from integer indices to label names.

    Returns:
        final_pred: The final predicted label (either index or label name).
        window_predictions: List of predictions for each window.
    """
    # Extract spectrograms using the same function used during dataset creation.
    spectrograms = FL.extract_spectrograms(audio_path, window_duration, sr, n_fft, hop_length)
    if len(spectrograms) == 0:
        print("No spectrograms were extracted from the audio file.")
        return None, []

    model.eval()
    window_predictions = []
    with torch.no_grad():
        for spec in spectrograms:
            spec_tensor = torch.tensor(spec, dtype=torch.float32)
            if transform:
                spec_tensor = transform(spec_tensor)
            # Add batch dimension: [1, 1, freq, time]
            spec_tensor = spec_tensor.unsqueeze(0)
            output = model(spec_tensor)
            # The predicted class is the argmax of the output logits.
            pred = torch.argmax(output, dim=1).item()
            window_predictions.append(pred)

    # Aggregate predictions via majority vote.
    final_pred_idx = max(set(window_predictions), key=window_predictions.count)

    # If a mapping is provided, convert the index to the corresponding label name.
    if idx_to_label:
        final_pred = idx_to_label.get(final_pred_idx, final_pred_idx)
    else:
        final_pred = final_pred_idx

    return final_pred, window_predictions


def train(model, dataloader):
    # Set up training components.
    learning_rate = 0.0005
    num_epochs = 15
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Simple training loop.
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(dataloader):
            # data: [batch, 1, 1025, 87], targets: [batch]
            outputs = model(data)  # Forward pass; outputs: [batch, num_classes]
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
