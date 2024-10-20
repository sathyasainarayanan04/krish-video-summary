import cv2 
import os

def extract_frames(video_path, output_folder, frame_rate=1):
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Error: Could not open video.")
        return

    fps = 10
    interval = int(fps / frame_rate)

    os.makedirs(output_folder, exist_ok=True)

    current_frame = 0
    extracted_frame = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        if current_frame % interval == 0:
            frame_path = os.path.join(output_folder, f"frame_{extracted_frame:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            extracted_frame += 1

        current_frame += 1

    video.release()
    print(f"Extracted {extracted_frame} frames.")

# Usage
video_path = '/content/drive/MyDrive/hemorrhoids/5b87e720-5dfa-42b8-95bc-e40202be7404.avi'
output_folder = '/content/drive/MyDrive/hemorrhoids2'
extract_frames(video_path, output_folder)
