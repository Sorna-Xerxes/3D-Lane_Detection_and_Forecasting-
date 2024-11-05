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
import datetime as dt
import time
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import networkx as nx

# Initialize timing
start_time = dt.datetime.now()
print('Start learning at {}'.format(str(start_time)))

# File paths
train_label_files = [
    r"C:\Users\91979\Desktop\Msc Course\Dissertation\Data\TuSimple\Trainset\label_data_0313.json",
    r"C:\Users\91979\Desktop\Msc Course\Dissertation\Data\TuSimple\Trainset\label_data_0531.json",
    r"C:\Users\91979\Desktop\Msc Course\Dissertation\Data\TuSimple\Trainset\label_data_0601.json"
]
test_label_file = r"C:\Users\91979\Desktop\Msc Course\Dissertation\Data\TuSimple\test_label.json"
evaluated_save_path = r"C:\Users\91979\Desktop\Msc Course\Dissertation\Data\Evaluated data"

# Ensure save path exists
os.makedirs(evaluated_save_path, exist_ok=True)

# Function to load image paths and lane annotations
def load_data(label_files):
    data = []
    for file_path in label_files:
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    return data

# Function to visualize lane annotations
def visualize_lane_annotations(data_samples, num_samples=5, dataset_type="Training", base_path=""):
    plt.figure(figsize=(15, num_samples * 5))
    for i, data_sample in enumerate(np.random.choice(data_samples, num_samples, replace=False)):
        image_path = os.path.join(base_path, data_sample['raw_file'])
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error: Image not found or failed to load: {image_path}")
            continue

        for lane in data_sample['lanes']:
            points = [(x, y) for x, y in zip(lane, data_sample['h_samples']) if x != -2]
            for point in points:
                cv2.circle(image, point, radius=5, color=(0, 255, 0), thickness=-1)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.subplot(num_samples, 1, i + 1)
        plt.imshow(image_rgb)
        plt.title(f"{dataset_type} Sample {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# 3D ResNet Backbone
class ResNet3DBackbone(nn.Module):
    def __init__(self):
        super(ResNet3DBackbone, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3))
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.LeakyReLU(inplace=True)
        
        # Adding additional layers to expand channels
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(128)
        self.relu2 = nn.LeakyReLU(inplace=True)
        
        self.conv3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(256)
        self.relu3 = nn.LeakyReLU(inplace=True)
        
        self.conv4 = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
        self.bn4 = nn.BatchNorm3d(512)
        self.relu4 = nn.LeakyReLU(inplace=True)
        
        # Adding dropout for regularization
        self.dropout = nn.Dropout3d(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        
        x = self.dropout(x)
        
        return x

# Enhanced 3D Encoder-Decoder PINet model with time-space split convolution
class EnhancedPINet(nn.Module):
    def __init__(self):
        super(EnhancedPINet, self).__init__()
        self.backbone = ResNet3DBackbone()
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(512, 256, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose3d(256, 128, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose3d(128, 64, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose3d(64, 32, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(inplace=True)
        )
        
        self.final_layer = nn.Conv2d(32, 1, kernel_size=1)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.decoder(x)
        x = x.squeeze(2)  # Remove temporal dimension
        x = torch.sigmoid(self.final_layer(x))
        return x

# Focal Loss Function to handle class imbalance and hard examples
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

# Line IOU Loss
class LineIOULoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(LineIOULoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        pred = pred.float()
        target = target.float()

        intersection = torch.sum(pred * target)
        union = torch.sum(pred) + torch.sum(target) - intersection

        iou = intersection / (union + 1e-6)

        loss = 1 - iou
        return loss if self.reduction == 'none' else loss.mean()

# Function to gather ROI around predicted and ground-truth lanes
def gather_roi(pred_mask, gt_mask, margin=10):
    roi_pred = np.zeros_like(pred_mask)
    roi_gt = np.zeros_like(gt_mask)

    # Find contours of predicted and ground truth lanes
    contours_pred, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_gt, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours_pred:
        x, y, w, h = cv2.boundingRect(contour)
        roi_pred[max(y-margin, 0):min(y+h+margin, roi_pred.shape[0]), max(x-margin, 0):min(x+w+margin, roi_pred.shape[1])] = pred_mask[max(y-margin, 0):min(y+h+margin, roi_pred.shape[0]), max(x-margin, 0):min(x+w+margin, roi_pred.shape[1])]

    for contour in contours_gt:
        x, y, w, h = cv2.boundingRect(contour)
        roi_gt[max(y-margin, 0):min(y+h+margin, roi_gt.shape[0]), max(x-margin, 0):min(x+w+margin, roi_gt.shape[1])] = gt_mask[max(y-margin, 0):min(y+h+margin, roi_gt.shape[0]), max(x-margin, 0):min(x+w+margin, roi_gt.shape[1])]

    return roi_pred, roi_gt

# Augmentation class for data processing
class Augmentation:
    def __init__(self, flip_prob=0.5, brightness_range=(0.8, 1.2), noise_std=0.01, rotation_angle=10):
        self.flip_prob = flip_prob
        self.brightness_range = brightness_range
        self.noise_std = noise_std
        self.rotation_angle = rotation_angle

    def __call__(self, image, mask):
        if random.random() < self.flip_prob:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        
        brightness_factor = random.uniform(*self.brightness_range)
        image = np.clip(image * brightness_factor, 0, 255)

        noise = np.random.normal(0, self.noise_std, image.shape)
        image = np.clip(image + noise * 255, 0, 255)

        angle = random.uniform(-self.rotation_angle, self.rotation_angle)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        image = cv2.warpAffine(image, M, (w, h))
        mask = cv2.warpAffine(mask, M, (w, h))

        return image, mask
    
def visualize_augmentations(image, mask):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Augmented Image")
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Augmented Mask")
    plt.show()
    
# Dataset class (now with augmentation)
class LaneDataset(Dataset):
    def __init__(self, data, base_path, augmentations=None):
        self.data = data
        self.base_path = base_path
        self.augmentations = augmentations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_sample = self.data[idx]
        image_path = os.path.join(self.base_path, data_sample['raw_file'])
        image = cv2.imread(image_path)
        image = cv2.resize(image, (256, 128))

        mask = np.zeros((128, 256), dtype=np.uint8)
        for lane in data_sample['lanes']:
            points = [(x * 256 // 1280, y * 128 // 720) for x, y in zip(lane, data_sample['h_samples']) if x != -2]
            for point in points:
                cv2.circle(mask, point, radius=3, color=1, thickness=-1)

        if self.augmentations:
            image, mask = self.augmentations(image, mask)

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask

# Load training data and visualize
train_data = load_data(train_label_files)
print(f"Total training samples: {len(train_data)}")
visualize_lane_annotations(train_data, num_samples=5, dataset_type="Training", base_path=r"C:\Users\91979\Desktop\Msc Course\Dissertation\Data\TuSimple\Trainset")

# Load test data and visualize
test_data = load_data([test_label_file])
print(f"Total test samples: {len(test_data)}")
visualize_lane_annotations(test_data, num_samples=5, dataset_type="Test", base_path=r"C:\Users\91979\Desktop\Msc Course\Dissertation\Data\TuSimple\Testset")

# Create training dataset and dataloader with augmentations
augmentations = Augmentation()
train_dataset = LaneDataset(train_data, r"C:\Users\91979\Desktop\Msc Course\Dissertation\Data\TuSimple\Trainset", augmentations=augmentations)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
print(f"Total number of batches per epoch: {len(train_loader)}")

# Model, optimizer, and loss function setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Adjust to use GPU if available
model = EnhancedPINet().to(device)  # Move the model to the correct device
optimizer = optim.Adam(model.parameters(), lr=0.1)  # Adjusted learning rate for better convergence
criterion = LineIOULoss()  # Using Line IOU Loss

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=1, patience=5):
    best_loss = float('inf')
    patience_counter = 0
    train_losses = []
    train_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        all_targets = []
        all_preds = []
        epoch_start_time = time.time()

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            # Add the extra dimension for depth (set it to 5 for temporal information)
            images = images.unsqueeze(2)  # Shape: (batch_size, channels, depth=5, height, width)

            optimizer.zero_grad()
            outputs = model(images)
            outputs_resized = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)
            loss = criterion(outputs_resized, masks)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            preds = (outputs_resized > 0.5).float()
            correct_predictions += (preds == masks).sum().item()
            total_predictions += masks.numel()
            
            all_targets.extend(masks.cpu().numpy().flatten())
            all_preds.extend(preds.cpu().numpy().flatten())

        avg_loss = running_loss / len(train_loader)
        train_accuracy = correct_predictions / total_predictions
        train_losses.append(avg_loss)
        train_accuracies.append(train_accuracy)

        epoch_elapsed_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.4f}, Time: {epoch_elapsed_time:.2f} seconds")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    # Plotting loss, accuracy, precision, and F1 score
    plt.figure(figsize=(16, 8))

    plt.subplot(2, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Epoch')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy vs Epoch')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Train the model
train_model(model, train_loader, criterion, optimizer)

# Function for Graph-Based Smoothing
def graph_based_smoothing(lane_points):
    G = nx.Graph()

    # Add lane points as nodes
    for i, point in enumerate(lane_points):
        G.add_node(i, pos=point)

    # Connect neighboring points with edges, weighted by their Euclidean distance
    for i in range(len(lane_points) - 1):
        G.add_edge(i, i + 1, weight=np.linalg.norm(np.array(lane_points[i]) - np.array(lane_points[i + 1])))

    # Find the shortest smooth path using Dijkstra or another algorithm
    path = nx.shortest_path(G, source=0, target=len(lane_points) - 1, weight='weight')

    # Retrieve the smoothed lane points
    smoothed_lane = [lane_points[i] for i in path]
    return smoothed_lane

def ransac_curve_fitting(points):
    # Ensure there are enough points for RANSAC (at least 3 points)
    if len(points) < 3:
        return points  # Return original points if insufficient for curve fitting

    points = np.array(points)
    X = points[:, 1].reshape(-1, 1)  # Use y-coordinates as input to fit the curve
    y = points[:, 0]  # x-coordinates as the target

    # RANSAC to fit a quadratic curve (polynomial of degree 2)
    model = make_pipeline(PolynomialFeatures(degree=2), RANSACRegressor(min_samples=2))

    # Fit the model (ensure enough samples are available)
    model.fit(X, y)

    # Fix the DeprecationWarning by extracting scalar values from min and max
    y_curve = np.arange(min(X).item(), max(X).item() + 1).reshape(-1, 1)
    x_curve = model.predict(y_curve)

    # Return the fitted points
    fitted_points = list(zip(x_curve.astype(int), y_curve.flatten().astype(int)))

    return fitted_points

def evaluate_model(model, test_data, device='cpu', save_path="", max_visualizations=5):
    eval_start_time = time.time()
    model.eval()  # Set the model to evaluation mode
    all_labels = []
    all_predictions = []
    min_dot_area = 50  # Minimum area to consider a detected region as a valid lane marking

    # Loop through all test data for evaluation purposes
    for i, data_sample in enumerate(test_data):
        # Load the test image
        image_path = os.path.join(r"C:\Users\91979\Desktop\Msc Course\Dissertation\Data\TuSimple\Testset", data_sample['raw_file'])
        image = cv2.imread(image_path)
        image_resized = cv2.resize(image, (256, 128))
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0

        with torch.no_grad():
            # Add the extra dimension for depth (set it to 1 for temporal dimension)
            image_tensor = image_tensor.unsqueeze(2)  # Shape: (batch_size, channels, depth=1, height, width)
            output = model(image_tensor.to(device))

        # Process the model output
        output = output.squeeze().cpu().numpy()
        output = (output > 0.5).astype(np.uint8)

        # Filter out small regions that could be noise
        num_labels, labels_im = cv2.connectedComponents(output)
        for label in range(1, num_labels):
            if np.sum(labels_im == label) < min_dot_area:
                output[labels_im == label] = 0

        # Convert the ground truth lanes to a binary mask for comparison
        labels_mask = np.zeros_like(output)
        for lane in data_sample['lanes']:
            points = [(x * 256 // 1280, y * 128 // 720) for x, y in zip(lane, data_sample['h_samples']) if x != -2]

            # Apply RANSAC curve fitting and smoothing
            fitted_points = ransac_curve_fitting(points)
            smoothed_points = graph_based_smoothing(fitted_points)

            # Draw smoothed points on the labels mask
            for point in smoothed_points:
                cv2.circle(labels_mask, point, radius=1, color=1, thickness=-1)

        # Flatten masks for comparison and accumulate labels/predictions
        labels_mask_flat = labels_mask.flatten()
        output_flat = output.flatten()

        if len(labels_mask_flat) == len(output_flat):
            all_labels.extend(labels_mask_flat)
            all_predictions.extend(output_flat)
        else:
            print(f"Warning: Mismatch in lengths for sample {i+1}. Label length: {len(labels_mask_flat)}, Prediction length: {len(output_flat)}")

    # Calculate evaluation metrics
    if len(all_labels) == len(all_predictions) and len(all_labels) > 0:
        # Compute the confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)

        # Compute percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        # Plot the confusion matrix with percentages
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_percent, annot=True, fmt=".2f", cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title('Confusion Matrix with Percentages')
        plt.show()

        accuracy = accuracy_score(all_labels, all_predictions)
        print(f"Overall Accuracy: {accuracy * 100:.2f}%")
    else:
        print(f"Error: Mismatch in label and prediction lengths or no data for accuracy calculation: {len(all_labels)} vs {len(all_predictions)}")
    
    eval_elapsed_time = time.time() - eval_start_time
    print(f"Evaluation Time: {eval_elapsed_time:.2f} seconds")

    # Now visualize only `max_visualizations` images
    for i, data_sample in enumerate(test_data[:max_visualizations]):
        print(f"Visualizing and saving sample {i+1}")
        image_path = os.path.join(r"C:\Users\91979\Desktop\Msc Course\Dissertation\Data\TuSimple\Testset", data_sample['raw_file'])
        image = cv2.imread(image_path)
        image_resized = cv2.resize(image, (256, 128))
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0

        with torch.no_grad():
            # Add the extra dimension for depth (set it to 1)
            image_tensor = image_tensor.unsqueeze(2)  # Shape: (batch_size, channels, depth=1, height, width)
            output = model(image_tensor.to(device))

        # Process the model output
        output = output.squeeze().cpu().numpy()
        output = (output > 0.5).astype(np.uint8)

        # Overlay lane detection on the original image
        lanes_overlay = cv2.resize(output, (image.shape[1], image.shape[0]))
        image[lanes_overlay > 0] = [0, 255, 0]

        # Save the visualized image
        save_file_path = os.path.join(save_path, f"evaluated_sample_{i+1}.jpg")
        cv2.imwrite(save_file_path, image)
        print(f"Saved evaluated image: {save_file_path}")

    eval_elapsed_time = time.time() - eval_start_time
    print(f"Evaluation Time: {eval_elapsed_time:.2f} seconds")

# Evaluate the model and save images
evaluate_model(model, test_data, device=device, save_path=evaluated_save_path, max_visualizations=5)

# Timing end
end_time = dt.datetime.now()
print('Stop learning {}'.format(str(end_time)))
elapsed_time = end_time - start_time
print('Elapsed learning {}'.format(str(elapsed_time)))









