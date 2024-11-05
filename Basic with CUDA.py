import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
from torch.cuda.amp import GradScaler, autocast  # For mixed precision training
import datetime as dt
import time

start_time = dt.datetime.now()
print('Start learning at {}'.format(str(start_time)))

# Path to the JSON label files (use a subset of the data)
label_files = [
    r"C:\Users\91979\Desktop\Msc Course\Dissertation\Data\TuSimple\Trainset\label_data_0313.json",
    r"C:\Users\91979\Desktop\Msc Course\Dissertation\Data\TuSimple\Trainset\label_data_0531.json"
]

# Function to load image paths and lane annotations
def load_data(label_files):
    data = []
    for file_path in label_files:
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    return data

# Load the training data
train_data = load_data(label_files)

# Example of how to load and visualize an image without lane annotations (Training data)
def visualize_train_image(data_sample):
    image_path = os.path.join(r"C:\Users\91979\Desktop\Msc Course\Dissertation\Data\TuSimple\Trainset", data_sample['raw_file'])
    image = cv2.imread(image_path)

    # Check if the image was loaded correctly
    if image is None:
        print(f"Error: Image not found or failed to load: {image_path}")
        return

    cv2.imshow('Original Image (No Lanes)', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Visualize 5 random training samples
for i in np.random.choice(len(train_data), 5, replace=False):
    visualize_train_image(train_data[i])

# Define the SimpleLaneNet model with reduced filters
class SimpleLaneNet(nn.Module):
    def __init__(self):
        super(SimpleLaneNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)  # Reduced filters
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # Reduced filters
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # Reduced filters
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Reduced filters
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(8, 1, kernel_size=1)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))
        x = F.relu(self.upconv1(x4))
        x = F.relu(self.upconv2(x))
        x = F.relu(self.upconv3(x))
        x = torch.sigmoid(self.final_conv(x))
        return x

# Instantiate the model
model = SimpleLaneNet()

# Define the LaneDataset class with reduced image resolution
class LaneDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_sample = self.data[idx]
        image_path = os.path.join(r"C:\Users\91979\Desktop\Msc Course\Dissertation\Data\TuSimple\Trainset", data_sample['raw_file'])
        image = cv2.imread(image_path)
        image = cv2.resize(image, (256, 128))  # Reduced resolution
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        mask = np.zeros((128, 256), dtype=np.uint8)  # Adjusted mask size
        for lane in data_sample['lanes']:
            points = [(x * 256 // 1280, y * 128 // 720) for x, y in zip(lane, data_sample['h_samples']) if x != -2]
            for point in points:
                cv2.circle(mask, point, radius=3, color=1, thickness=-1)  # Reduced point size
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask

# Create the dataset and dataloader with reduced batch size
train_dataset = LaneDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)  # Reduced batch size

# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Initialize mixed precision training scaler
scaler = GradScaler()

# Training loop with early stopping
num_epochs = 10
best_loss = float('inf')
patience = 2
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to('cpu'), masks.to('cpu')
        
        optimizer.zero_grad()
        with autocast():  # Mixed precision training
            outputs = model(images)
            outputs_resized = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)
            loss = criterion(outputs_resized, masks)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # Early stopping logic
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping triggered")
        break

# Load test data
test_label_file = r"C:\Users\91979\Desktop\Msc Course\Dissertation\Data\TuSimple\Testset\test_label.json"
with open(test_label_file, 'r') as f:
    test_data = [json.loads(line.strip()) for line in f]

# Switch the model to evaluation mode
model.eval()

def evaluate_model(model, test_data, device='cpu'):
    all_labels = []
    all_predictions = []
    for i, data_sample in enumerate(test_data[:5]):  # Get 5 samples for simplicity
        image_path = os.path.join(r"C:\Users\91979\Desktop\Msc Course\Dissertation\Data\TuSimple\Testset", data_sample['raw_file'])
        image = cv2.imread(image_path)
        image_resized = cv2.resize(image, (256, 128))  # Reduced resolution
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0

        with torch.no_grad():
            output = model(image_tensor.to(device))

        output = output.squeeze().cpu().numpy()
        output = (output > 0.5).astype(np.uint8)

        all_labels.extend(np.array(data_sample['lanes']).flatten())
        all_predictions.extend(output.flatten())

        lanes_overlay = cv2.resize(output, (image.shape[1], image.shape[0]))
        image[lanes_overlay > 0] = [0, 255, 0]

        cv2.imshow(f"Detected Lanes - Sample {i+1}", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")

# Evaluate the model
evaluate_model(model, test_data)


end_time = dt.datetime.now() 
print('Stop learning {}'.format(str(end_time)))
elapsed_time= end_time - start_time
print('Elapsed learning {}'.format(str(elapsed_time)))
