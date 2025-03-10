Audio Spectrogram Classification with CNN-LSTM

This repository contains a project that converts audio files into spectrograms using a sliding window approach (with 20% overlap) and trains a CNN-LSTM model to associate each spectrogram with its corresponding filename label. The project includes:

    Data Preprocessing:
    Extraction of spectrograms from audio files using Librosa, with options for data augmentation via SpecAugment.

    Model Architectures:
    Implementation of a CNN-LSTM model and a CNN model variant (for experimental purposes) to learn the relationship between spectrogram features and file labels.

    Training and Inference:
    Training routines with adjustable hyperparameters (including learning rate scheduling, weight decay, and gradient clipping) to improve convergence. Inference functions are provided to classify new audio files.

    Intermediate Feature Visualization:
    Code to attach forward hooks to convolutional layers and visualize intermediate feature maps from a real spectrogram drawn from the dataloader. This helps in debugging and understanding the model's internal representations.

Feel free to explore, modify, and extend the project for your own audio classification tasks or research purposes. Contributions and feedback are welcome!
