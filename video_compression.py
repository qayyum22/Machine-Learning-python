import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def compress_video(input_path, output_path, target_size_mb=10):
    # Read the input video
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate target bitrate (in bits per second)
    target_size_bits = target_size_mb * 8 * 1024 * 1024
    duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)
    target_bitrate = int(target_size_bits / duration)
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Write the frame to the output video
        out.write(frame)
    
    # Release everything
    cap.release()
    out.release()
    
    print(f"Video compressed and saved to {output_path}")
    return output_path

def extract_features(video_path):
    cap = cv2.VideoCapture(video_path)
    features = []
    labels = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate average pixel intensity as a simple feature
        avg_intensity = np.mean(gray)
        features.append(avg_intensity)
        
        # Assign a simple label based on intensity (this is just for demonstration)
        label = 1 if avg_intensity > 127 else 0
        labels.append(label)
    
    cap.release()
    return np.array(features).reshape(-1, 1), np.array(labels)

def train_model(features, labels):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Train a Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Make predictions and calculate accuracy
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model trained with accuracy: {accuracy:.2f}")
    return clf

def main():
    input_video = "input_video.mp4"  # Replace with your input video path
    compressed_video = "compressed_video.mp4"
    target_size_mb = 10  # Target size in MB
    
    # Compress the video
    compressed_path = compress_video(input_video, compressed_video, target_size_mb)
    
    # Extract features from the compressed video
    features, labels = extract_features(compressed_path)
    
    # Train a model using the extracted features
    model = train_model(features, labels)

if __name__ == "__main__":
    main()