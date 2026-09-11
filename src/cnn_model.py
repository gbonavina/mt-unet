import torch
import torch.nn as nn
import torch.nn.functional as F


def _resolve_device(device):
    if device is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        return 'cpu'
    return device


class CNN(nn.Module):
    def __init__(
        self,
        input_channel,
        output_channels=None,
        kernel_size=3,
        stride=1,
        padding=1,
        n_classes=2,
        dropout=0.5,
    ):
        super(CNN, self).__init__()

        if output_channels is None:
            output_channels = [16, 32, 64]

        self.n_convs = len(output_channels)

        self.conv_layers = nn.ModuleList()
        self.pooling_layers = nn.ModuleList()
        self.batchnorm_layers = nn.ModuleList()

        for i in range(self.n_convs):
            in_ch = input_channel if i == 0 else output_channels[i - 1]
            self.conv_layers.append(nn.Conv2d(in_ch, output_channels[i], kernel_size, stride, padding))
            self.pooling_layers.append(nn.MaxPool2d(2))
            self.batchnorm_layers.append(nn.BatchNorm2d(output_channels[i]))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.fc_layer = nn.Linear(output_channels[-1], n_classes)

    def forward(self, x):
        for i in range(self.n_convs):
            x = self.conv_layers[i](x)
            x = self.batchnorm_layers[i](x)
            x = F.relu(x)
            x = self.pooling_layers[i](x)

        x = self.gap(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc_layer(x)
        return x

    def fit(self, train_loader, val_loader, epochs=10, lr=0.001, weight_decay=1e-5, device=None):
        device = _resolve_device(device)
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
        }

        for epoch in range(epochs):
            self.train()
            train_loss = 0
            train_correct = 0
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device).long()
                optimizer.zero_grad()
                output = self(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_correct += (output.argmax(dim=1) == labels).sum().item()

            self.eval()
            val_loss = 0
            val_correct = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device).long()
                    output = self(images)
                    loss = criterion(output, labels)
                    val_loss += loss.item()
                    val_correct += (output.argmax(dim=1) == labels).sum().item()

            n_train_batches = max(len(train_loader), 1)
            n_val_batches = max(len(val_loader), 1)
            train_loss /= n_train_batches
            train_accuracy = train_correct / max(len(train_loader.dataset), 1)
            val_loss /= n_val_batches
            val_accuracy = val_correct / max(len(val_loader.dataset), 1)

            history['train_loss'].append(train_loss)
            history['train_accuracy'].append(train_accuracy)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)

            print(
                f'Epoch {epoch + 1}/{epochs}, '
                f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
                f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}'
            )

        return history

    def predict(self, images, device=None):
        device = _resolve_device(device)
        self.to(device)
        self.eval()
        with torch.no_grad():
            images = images.to(device)
            output = self(images)
            return output.argmax(dim=1)
