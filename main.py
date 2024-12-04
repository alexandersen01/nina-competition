import os
import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pytesseract
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Function to sanitize the labels
def sanitize_label(label):
    # Only keep alphanumeric characters (0-9, A-Z)
    allowed_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "".join([char for char in label if char in allowed_chars])


class RingCodeDataset(Dataset):
    def __init__(self, image_paths, labels, max_length=5, transform=None):
        self.labels = [
            str(label)[:max_length].ljust(max_length)[:max_length] for label in labels
        ]
        self.image_paths = image_paths
        self.max_length = max_length  # Add this line to initialize max_length

        # label encoder
        unique_chars = sorted(set("".join(self.labels)))
        self.label_encoder = {
            char: idx for idx, char in enumerate(unique_chars, start=1)
        }
        self.label_encoder[""] = 0  # padding
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((64, 256)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Open and transform image
        image = Image.open(self.image_paths[idx]).convert("L")
        image = self.transform(image)

        # Encode label as tensor of indices
        label_indices = [self.label_encoder.get(char, 0) for char in self.labels[idx]]
        label_indices = label_indices[:self.max_length]  # Truncate to max_length
        label_indices += [0] * (self.max_length - len(label_indices))  # Pad if necessary
        label_tensor = torch.tensor(label_indices, dtype=torch.long)

        # Calculate input length (number of timesteps for RNN input)
        input_length = image.shape[2] // 4

        # Length of the target sequence (number of characters in the label)
        target_length = len(self.labels[idx])

        return image, label_tensor, input_length, target_length

    def get_label_encoder(self):
        return self.label_encoder



class RingCodeOCRNet(nn.Module):
    def __init__(self, num_classes, cnn_output_size=128, max_length=5):
        super(RingCodeOCRNet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.rnn = nn.Sequential(
            nn.LSTM(input_size=cnn_output_size, hidden_size=512, num_layers=2, batch_first=True),
        )

        self.fc = nn.Linear(512, num_classes)
        self.max_length = max_length

    def forward(self, x):
        # CNN layers
        x = self.cnn(x)
        # Determine input length (width) after CNN feature extractor
        input_length = x.shape[2]  # Width of the image after CNN
        
        # Flatten the output for RNN input (batch_size, input_length, feature_size)
        x = x.flatten(2).permute(0, 2, 1)  # (batch_size, input_length, feature_size)

        # Pass through RNN
        rnn_out, _ = self.rnn(x)

        # Output layers
        out = self.fc(rnn_out)
        return out





def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (256, 64))
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return image


def prepare_datasets(datasets_paths):
    all_images = []
    all_labels = []

    for dataset_path in datasets_paths:
        labels_csv = os.path.join(dataset_path, "ringcodes.csv")
        labels_df = pd.read_csv(labels_csv, sep="|")

        images_dir = os.path.join(dataset_path, "images")

        for _, row in labels_df.iterrows():
            filename = row["filename"]
            image_path = os.path.join(images_dir, filename)
            ring_code = row["code"]

            if os.path.exists(image_path):
                all_images.append(image_path)
                all_labels.append(ring_code)

    print(f"Total images with labels: {len(all_images)}")

    # Sanitize the labels before encoding
    all_labels = [sanitize_label(label) for label in all_labels]

    return all_images, all_labels


def train_ring_code_ocr(train_loader, val_loader, device):
    model = RingCodeOCRNet(num_classes=39)
    model.to(device)
    
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        for images, labels, input_lengths, target_lengths in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, labels, input_lengths, target_lengths)
            
            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

    return model



def evaluate_model(model, val_loader, device="cuda"):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    print(classification_report(true_labels, predictions))


def main():
    print('hello')
    datasets = [
        "ringreadingcompetition/datasets/lyngoy",
        "ringreadingcompetition/datasets/rf",
        "ringreadingcompetition/datasets/ringmerkingno",
    ]

    images, labels = prepare_datasets(datasets)

    # Create the label encoder based on sanitized labels
    label_encoder = {char: idx for idx, char in enumerate(set("".join(labels)))}

    # Encode the labels
    encoded_labels = [
        torch.tensor([label_encoder[char] for char in label]) for label in labels
    ]

    X_train, X_val, y_train, y_val = train_test_split(
        images, encoded_labels, test_size=0.2, random_state=42
    )

    train_dataset = RingCodeDataset(X_train, y_train)
    val_dataset = RingCodeDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = train_ring_code_ocr(train_loader, val_loader, device)
    evaluate_model(model, val_loader, device)


if __name__ == "__main__":
    main()
