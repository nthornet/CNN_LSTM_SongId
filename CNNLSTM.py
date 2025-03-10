import FileLoading as FL
import HybridModel as HM
import NF_EEG as NF
import torch
import logging

AUDIO_DIRECTORY = 'Data'  # Change this to your actual path.
WINDOW_DURATION = 2.0  # seconds per window
MODEL_PATH = "Model/model.pth"

logger = logging.getLogger(__name__)


def visualize_cnn_features():
    X, y = FL.create_dataset(AUDIO_DIRECTORY, window_duration=WINDOW_DURATION)
    # Use a transformation without augmentation for visualization.
    transform = FL.SpecAugmentationTransform(augment=False)
    dataloader = FL.create_dataloader(X, y, batch_size=16, shuffle=True, transform=transform)

    # --- Extract a real spectrogram from the dataloader ---
    # Get one batch and select the first sample.
    for data, labels in dataloader:
        # data shape is (batch, channels, height, width)
        sample = data[0].unsqueeze(0)  # add a batch dimension -> shape: (1, channels, height, width)
        break

    # --- Initialize the model using the same input shape as the sample ---
    _, channels, freq, time = sample.shape
    input_size = (freq, time)
    num_classes = len(set(y))
    model = NF.CNNModel(num_classes=num_classes, input_channels=channels, input_size=input_size)

    # --- Visualize intermediate features using the real spectrogram ---
    NF.visualize_intermediate_features(model, sample)


def training_NF(model_path="Model/model_NF.pth"):
    X, y = FL.create_dataset(AUDIO_DIRECTORY, window_duration=WINDOW_DURATION)
    logger.debug(f"Created dataset with {len(X)} spectrograms and {len(set(y))} unique labels.")
    # Create DataLoader using the dataset and transform.
    transform = FL.SpecAugmentationTransform(augment=False)
    dataloader = FL.create_dataloader(X, y, batch_size=16, shuffle=True, transform=transform)
    dataset = dataloader.dataset
    idx_to_label = {v: k for k, v in dataset.label_to_idx.items()}
    logger.debug(f"Index to Label mapping:{idx_to_label}", )

    # Create an instance of the CNN-LSTM model.
    num_classes = len(set(y))
    model = NF.CNNModel(num_classes=num_classes, input_channels=1, input_size=(X[0].shape[0], X[0].shape[1]))
    NF.train(model, dataloader, 15)
    torch.save(model.state_dict(), model_path)
    return model, idx_to_label


def training(model_path="Model/model.pth"):
    # Define your audio directory and window duration.
    # Create dataset from audio files.
    X, y = FL.create_dataset(AUDIO_DIRECTORY, window_duration=WINDOW_DURATION)
    logger.debug(f"Created dataset with {len(X)} spectrograms and {len(set(y))} unique labels.")
    # Create DataLoader using the dataset and transform.
    transform = FL.SpecAugmentationTransform(augment=True)
    dataloader = FL.create_dataloader(X, y, batch_size=16, shuffle=True, transform=transform)
    dataset = dataloader.dataset
    idx_to_label = {v: k for k, v in dataset.label_to_idx.items()}
    logger.debug(f"Index to Label mapping:{idx_to_label}", )

    # Create an instance of the CNN-LSTM model.
    num_classes = len(set(y))
    model = HM.CNNLSTM(num_classes=num_classes)
    HM.train(model, dataloader)
    torch.save(model.state_dict(), model_path)
    return model, idx_to_label


def inference_nf():
    model_path = "Model/model_NF.pth"
    X, y = FL.create_dataset(AUDIO_DIRECTORY, window_duration=WINDOW_DURATION)
    logger.debug(f"Created dataset with {len(X)} spectrograms and {len(set(y))} unique labels.")

    # Create DataLoader using the dataset and transform.
    transform = FL.SpecAugmentationTransform(augment=True)
    dataloader = FL.create_dataloader(X, y, batch_size=16, shuffle=True, transform=transform)
    dataset = dataloader.dataset
    idx_to_label = {v: k for k, v in dataset.label_to_idx.items()}
    logger.debug(f"Index to Label mapping:{idx_to_label}", )

    # Use the same input size as in training.
    num_classes = len(set(y))
    input_size = (X[0].shape[0], X[0].shape[1])
    model = NF.CNNModel(num_classes=num_classes, input_channels=1, input_size=input_size)

    # Load the trained state dictionary.
    model.load_state_dict(torch.load(model_path))

    inference_transform = FL.SpecAugmentationTransform(augment=False)
    final_prediction, predictions_per_window = HM.classify_new_audio(
        model,
        "Data/Sub Basics - Temple of Sound 002 - Sub Basics - 01 Sub Basics - Nomad.flac",
        window_duration=WINDOW_DURATION,
        transform=inference_transform,
        idx_to_label=idx_to_label,
    )

    return final_prediction


def inference(idx_to_label, model_path="Model/model.pth"):
    # Define your audio directory and window duration.
    # Create dataset from audio files.
    X, y = FL.create_dataset(AUDIO_DIRECTORY, window_duration=WINDOW_DURATION)
    logger.debug(f"Created dataset with {len(X)} spectrograms and {len(set(y))} unique labels.")
    # Create DataLoader using the dataset and transform.
    transform = FL.SpecAugmentationTransform(augment=True)
    dataloader = FL.create_dataloader(X, y, batch_size=16, shuffle=True, transform=transform)
    dataset = dataloader.dataset
    idx_to_label = {v: k for k, v in dataset.label_to_idx.items()}
    logger.debug(f"Index to Label mapping:{idx_to_label}", )

    # Create an instance of the CNN-LSTM model.
    num_classes = len(set(y))
    model = HM.CNNLSTM(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    inference_transform = FL.SpecAugmentationTransform(augment=False)
    final_prediction, predictions_per_window = HM.classify_new_audio(
        model,
        "Data/Al Wooton_ Philo.wav",
        window_duration=WINDOW_DURATION,
        transform=inference_transform,
        idx_to_label=idx_to_label,
    )

    return final_prediction


if __name__ == '__main__':
    logging.basicConfig(filename='Shazam.log', level=logging.DEBUG)
    #visualize_cnn_features()
    #model, idx_to_label = training_NF()
    #model, idx_to_label = training()
    prediction = inference_nf()
    print(f"Final Prediction: {prediction}")
