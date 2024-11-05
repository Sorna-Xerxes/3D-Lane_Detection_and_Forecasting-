import os
import json
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Paths to the data
train_clips_dir = r"C:\Users\91979\Desktop\Msc Course\Dissertation\Data\TUSimple\Trainset\clips"
test_clips_dir = r"C:\Users\91979\Desktop\Msc Course\Dissertation\Data\TUSimple\Testset\clips"
train_json_files = [
    r"C:\Users\91979\Desktop\Msc Course\Dissertation\Data\TUSimple\Trainset\label_data_0313.json",
    r"C:\Users\91979\Desktop\Msc Course\Dissertation\Data\TUSimple\Trainset\label_data_0531.json",
    r"C:\Users\91979\Desktop\Msc Course\Dissertation\Data\TUSimple\Trainset\label_data_0601.json"
]
test_json_file = r"C:\Users\91979\Desktop\Msc Course\Dissertation\Data\TUSimple\Testset\test_tasks_0627.json"

def count_clips_in_directory(clips_dir):
    total_clips = 0
    for subdir in os.listdir(clips_dir):
        subdir_path = os.path.join(clips_dir, subdir)
        if os.path.isdir(subdir_path):
            num_clips = len(os.listdir(subdir_path))
            total_clips += num_clips
    return total_clips

def read_json_files(json_files):
    data = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            for line in f:
                data.append(json.loads(line))
    return data

def read_test_json_file(json_file):
    data = []
    with open(json_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def display_random_frames(data, clips_dir, num_frames=5):
    selected_data = random.sample(data, num_frames)
    
    for item in selected_data:
        raw_file_path = item['raw_file'].replace("clips/", "")
        img_path = os.path.join(clips_dir, raw_file_path)
        
        if not os.path.exists(img_path):
            print(f"Image path {img_path} does not exist.")
            continue
        
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"Failed to load image at {img_path}")
            continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_with_labels = image_rgb.copy()
        
        h_samples = item['h_samples']
        lanes = item['lanes']
        
        for lane in lanes:
            points = [(x, y) for x, y in zip(lane, h_samples) if x != -2]
            for point in points:
                cv2.circle(image_with_labels, point, 5, (255, 0, 0), -1)
        
        # Display the image with and without labels
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(image_rgb)
        plt.title("Frame without Labels")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(image_with_labels)
        plt.title("Frame with Labels")
        plt.axis('off')
        
        plt.show()

def main():
    # Count the clips in the training directory
    total_train_clips = count_clips_in_directory(train_clips_dir)
    print(f"Number of videoclips in training set: {total_train_clips}")

    # Read JSON files for training data
    train_data = read_json_files(train_json_files)
    total_train_labels = len(train_data)
    print(f"Number of labeled videoclips in training set: {total_train_labels}")
    
    # Display random frames from training set with and without labels
    print("Displaying random frames from training set:")
    display_random_frames(train_data, train_clips_dir)
    
    # Count the clips in the testing directory
    total_test_clips = count_clips_in_directory(test_clips_dir)
    print(f"Number of videoclips in testing set: {total_test_clips}")

    # Read JSON file for testing data
    test_data = read_test_json_file(test_json_file)
    total_test_labels = len(test_data)
    print(f"Number of videoclips in testing set: {total_test_labels}")
    
    # Display random frames from testing set with and without labels
    print("Displaying random frames from testing set:")
    display_random_frames(test_data, test_clips_dir)

if __name__ == "__main__":
    main()
