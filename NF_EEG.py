import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

class CNNModel(nn.Module):
    def __init__(self, num_classes=10, input_channels=64, input_size=(1025, 431)):
        """
        Args:
            num_classes (int): Number of output classes.
            input_channels (int): Number of channels in the input.
            input_size (tuple): Height and width of the input spectrogram (e.g., (1025, 431)).
        """
        super(CNNModel, self).__init__()

        # Block 1: Conv2D -> BatchNorm -> DepthwiseConv2D -> ELU -> AvgPool -> Dropout
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=8, kernel_size=(3, 1),
                      stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(8),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 1), groups=8, padding=(1, 0)),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Dropout(p=0.5)
        )

        # Block 2: Conv2D -> BatchNorm -> SeparableConv2D -> ELU -> AvgPool -> Dropout
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 1), groups=16, padding=(1, 0)),  # Depthwise
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 1)),  # Pointwise
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Dropout(p=0.5)
        )

        # Block 3: Conv2D -> BatchNorm -> DepthwiseConv2D -> ELU -> AvgPool -> Dropout
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 1), groups=16, padding=(1, 0)),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Dropout(p=0.5)
        )

        # Block 4: Conv2D -> BatchNorm -> SeparableConv2D -> ELU -> AvgPool -> Dropout
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 1), groups=16, padding=(1, 0)),  # Depthwise
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 1)),  # Pointwise
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Dropout(p=0.5)
        )

        self.flatten = nn.Flatten()

        # Determine the flattened feature size using a dummy input
        dummy_input = torch.zeros(1, input_channels, input_size[0], input_size[1])
        dummy_out = self.block1(dummy_input)
        dummy_out = self.block2(dummy_out)
        dummy_out = self.block3(dummy_out)
        dummy_out = self.block4(dummy_out)
        flattened_size = dummy_out.view(1, -1).size(1)

        # Initialize the fully-connected layer using the computed flattened size.
        self.fc = nn.Linear(flattened_size, num_classes)

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, input_channels, height, width)
        Returns:
            torch.Tensor: Output probabilities of shape (batch_size, num_classes)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def train(model, dataloader, num_epochs = 15):
    learning_rate = 0.001
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()  # Set the model to training mode.
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (data, targets) in enumerate(dataloader):
            # data: shape (batch, channels, 1025, 431)
            outputs = model(data)  # Forward pass.
            loss = criterion(outputs, targets)

            optimizer.zero_grad()  # Clear gradients.
            loss.backward()  # Backpropagation.
            optimizer.step()  # Update weights.

            running_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")


def get_conv_layers(model):
    """
    Recursively collects all convolutional layers in the model.

    Args:
        model (nn.Module): The model to search.

    Returns:
        dict: A dictionary mapping layer names to the module objects.
    """
    conv_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers[name] = module
    return conv_layers


def visualize_intermediate_features(model, input_tensor):
    """
    Runs a forward pass through the model while capturing the outputs
    of all convolutional layers, then visualizes each layer’s feature maps.

    Args:
        model (nn.Module): The neural network model.
        input_tensor (torch.Tensor): A sample input tensor of shape
                                     [batch, channels, height, width].
    """
    # Retrieve all Conv2d layers from the model.
    conv_layers = get_conv_layers(model)
    features = {}

    # Define a hook function that will save the output of a layer.
    def hook_fn(module, input, output):
        features[module] = output.detach().cpu()

    # Register the hook on each convolutional layer.
    hooks = []
    for name, module in conv_layers.items():
        hook = module.register_forward_hook(hook_fn)
        hooks.append(hook)

    # Ensure the model is in evaluation mode and run a forward pass.
    model.eval()
    with torch.no_grad():
        _ = model(input_tensor)

    # Remove all hooks to avoid side effects.
    for hook in hooks:
        hook.remove()

    # Visualize the feature maps for each convolutional layer.
    for module, feat in features.items():
        # feat is of shape: (batch_size, num_channels, height, width)
        num_channels = feat.shape[1]
        num_cols = min(num_channels, 8)  # number of columns in the grid
        num_rows = (num_channels + num_cols - 1) // num_cols  # calculate rows
        plt.figure(figsize=(num_cols * 2, num_rows * 2))
        for i in range(num_channels):
            plt.subplot(num_rows, num_cols, i + 1)
            plt.imshow(feat[0, i, :, :], cmap='viridis')
            plt.title(f'Ch {i}')
            plt.axis('off')
        plt.suptitle(f"Feature maps from layer: {module}")
        plt.tight_layout()
        plt.show()


# Example usage
def example():
    # Initialize the model.
    # Change input_channels if your spectrogram is single-channel.
    model = CNNModel(num_classes=10, input_channels=64, input_size=(1025, 431))
    print(model)

    # Test with a dummy input (using batch size = 2)
    dummy_input = torch.randn(2, 64, 1025, 431)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # Expected shape: (2, 10)

