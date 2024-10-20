import os
import cv2
from PIL import Image
import torch
from collections import Counter
import matplotlib.pyplot as plt
class_description = {
    "lower-gi-tract/anatomical-landmarks/cecum": "The cecum is the beginning of the large intestine and is a critical anatomical landmark in the lower gastrointestinal tract. It plays a role in absorbing fluids and salts that remain after intestinal digestion and absorption.",
    "lower-gi-tract/anatomical-landmarks/ileum": "The ileum is the final section of the small intestine, where absorption of vitamin B12 and bile salts occurs. It is a key anatomical landmark in the lower gastrointestinal tract.",
    "lower-gi-tract/anatomical-landmarks/retroflex-rectum": "Retroflex view of the rectum is an endoscopic technique used to inspect the rectum more thoroughly, especially for polyps or other abnormalities.",
    "lower-gi-tract/pathological-findings/hemorrhoids": "Hemorrhoids are swollen veins in the lower rectum or anus, often causing discomfort, itching, or bleeding. They are a common pathological finding in the lower gastrointestinal tract.",
    "lower-gi-tract/pathological-findings/polyps": "Polyps are abnormal growths of tissue that form on the lining of the colon or rectum. They can vary in size and have the potential to develop into colorectal cancer if left untreated.",
    "lower-gi-tract/pathological-findings/ulcerative-colitis-grade-0-1": "Ulcerative Colitis Grade 0-1 indicates minimal to mild inflammation of the colon, characterized by superficial ulcerations and erythema. It is an early stage of the disease in the lower gastrointestinal tract.",
    "lower-gi-tract/pathological-findings/ulcerative-colitis-grade-1": "Ulcerative Colitis Grade 1 is marked by mild inflammation in the colon with superficial ulcerations and erythema. It is an early stage of ulcerative colitis.",
    "lower-gi-tract/pathological-findings/ulcerative-colitis-grade-1-2": "Ulcerative Colitis Grade 1-2 represents moderate inflammation of the colon, with more pronounced ulcerations and erythema. This stage indicates a progression of the disease.",
    "lower-gi-tract/pathological-findings/ulcerative-colitis-grade-2": "Ulcerative Colitis Grade 2 involves moderate inflammation and ulceration in the colon. It is a more advanced stage of the disease, requiring careful monitoring.",
    "lower-gi-tract/pathological-findings/ulcerative-colitis-grade-2-3": "Ulcerative Colitis Grade 2-3 indicates severe inflammation with extensive ulceration and erythema. This stage of the disease requires aggressive treatment.",
    "lower-gi-tract/pathological-findings/ulcerative-colitis-grade-3": "Ulcerative Colitis Grade 3 is characterized by severe inflammation, deep ulcerations, and extensive tissue damage in the colon. It is the most advanced stage of the disease.",
    "lower-gi-tract/quality-of-mucosal-views/bbps-0-1": "The Boston Bowel Preparation Scale (BBPS) score of 0-1 indicates poor bowel preparation, with a significant amount of stool obstructing the view during a colonoscopy, affecting the quality of mucosal views.",
    "lower-gi-tract/quality-of-mucosal-views/bbps-2-3": "The Boston Bowel Preparation Scale (BBPS) score of 2-3 indicates fair to good bowel preparation, with partial to clear views of the mucosa during a colonoscopy, allowing for a more thorough examination.",
    "lower-gi-tract/quality-of-mucosal-views/impacted-stool": "Impacted stool refers to a large, hard mass of stool that is stuck in the colon or rectum. It can obscure the view during endoscopic procedures, compromising the quality of mucosal views.",
    "lower-gi-tract/therapeutic-interventions/dyed-lifted-polyps": "Dyed lifted polyps refer to polyps that have been injected with a dye solution to lift them away from the mucosal layer during an endoscopic procedure, facilitating their removal.",
    "lower-gi-tract/therapeutic-interventions/dyed-resection-margins": "Dyed resection margins refer to the edges of tissue that have been stained with dye during an endoscopic procedure to ensure that the entire lesion or polyp has been removed.",
    "upper-gi-tract/anatomical-landmarks/pylorus": "The pylorus is the opening from the stomach into the duodenum (first part of the small intestine). It is an important anatomical landmark in the upper gastrointestinal tract, controlling the passage of stomach contents.",
    "upper-gi-tract/anatomical-landmarks/retroflex-stomach": "Retroflex view of the stomach is an endoscopic technique used to inspect the upper part of the stomach, especially for detecting abnormalities such as polyps, ulcers, or tumors.",
    "upper-gi-tract/anatomical-landmarks/z-line": "The Z-line, or squamocolumnar junction, is where the esophagus meets the stomach. It is an important anatomical landmark in the upper gastrointestinal tract, often inspected for signs of Barrett's esophagus or other conditions.",
    "upper-gi-tract/pathological-findings/barretts": "Barrett's esophagus is a condition in which the lining of the esophagus changes to resemble the lining of the stomach, increasing the risk of esophageal cancer. It is a significant pathological finding in the upper gastrointestinal tract.",
    "upper-gi-tract/pathological-findings/barretts-short-segment": "Barrett's esophagus with a short segment refers to a limited area of the esophagus where the lining has changed to resemble the stomach lining. This condition increases the risk of developing esophageal cancer.",
    "upper-gi-tract/pathological-findings/esophagitis-a": "Esophagitis Grade A indicates mild inflammation of the esophagus, often due to acid reflux. It is characterized by small, isolated areas of erosion in the lining of the esophagus.",
    "upper-gi-tract/pathological-findings/esophagitis-b-d": "Esophagitis Grades B-D represent progressively severe inflammation and damage to the esophagus lining, often due to acid reflux, with Grade D being the most severe, involving extensive erosion."
}

def extract_frames(video_path, output_folder, frame_rate=1):
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Error: Could not open video.")
        return

    fps = video.get(cv2.CAP_PROP_FPS)
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

# Define the class-to-index mapping
class_to_index = {
    'lower-gi-tract/anatomical-landmarks/cecum': 0,
    'lower-gi-tract/anatomical-landmarks/ileum': 1,
    'lower-gi-tract/anatomical-landmarks/retroflex-rectum': 2,
    'lower-gi-tract/pathological-findings/hemorrhoids': 3,
    'lower-gi-tract/pathological-findings/polyps': 4,
    'lower-gi-tract/pathological-findings/ulcerative-colitis-grade-0-1': 5,
    'lower-gi-tract/pathological-findings/ulcerative-colitis-grade-1': 6,
    'lower-gi-tract/pathological-findings/ulcerative-colitis-grade-1-2': 7,
    'lower-gi-tract/pathological-findings/ulcerative-colitis-grade-2': 8,
    'lower-gi-tract/pathological-findings/ulcerative-colitis-grade-2-3': 9,
    'lower-gi-tract/pathological-findings/ulcerative-colitis-grade-3': 10,
    'lower-gi-tract/quality-of-mucosal-views/bbps-0-1': 11,
    'lower-gi-tract/quality-of-mucosal-views/bbps-2-3': 12,
    'lower-gi-tract/quality-of-mucosal-views/impacted-stool': 13,
    'lower-gi-tract/therapeutic-interventions/dyed-lifted-polyps': 14,
    'lower-gi-tract/therapeutic-interventions/dyed-resection-margins': 15,
    'upper-gi-tract/anatomical-landmarks/pylorus': 16,
    'upper-gi-tract/anatomical-landmarks/retroflex-stomach': 17,
    'upper-gi-tract/anatomical-landmarks/z-line': 18,
    'upper-gi-tract/pathological-findings/barretts': 19,
    'upper-gi-tract/pathological-findings/barretts-short-segment': 20,
    'upper-gi-tract/pathological-findings/esophagitis-a': 21,
    'upper-gi-tract/pathological-findings/esophagitis-b-d': 22
}

# Reverse the dictionary to map indexes to class names
index_to_class = {index: cls for cls, index in class_to_index.items()}

def predict_frame(model, frame_path, transform):
    image = Image.open(frame_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        _, predicted_index = torch.max(output, 1)
    
    predicted_index_value = predicted_index.item()
    # Check if predicted index is in index_to_class
    if predicted_index_value in index_to_class:
        predicted_class = index_to_class[predicted_index_value]
    else:
        print(f"Warning: Index {predicted_index_value} not found in index_to_class.")
        predicted_class = 'Unknown'
    
    return predicted_class

def group_consecutive_frames(predictions, fps):
    """
    Group consecutive frames with the same predicted label and calculate the time range.
    """
    if not predictions:
        return []

    grouped_results = []
    current_label = None
    start_frame = None

    for frame_index, predicted_label in sorted(predictions.items()):
        if predicted_label != current_label:
            if current_label is not None:
                # End of a group, calculate the time range
                end_frame = frame_index - 1
                start_time = start_frame / fps
                end_time = end_frame / fps
                grouped_results.append({
                    'label': current_label,
                    'start_time': start_time,
                    'end_time': end_time
                })
            # Start a new group
            current_label = predicted_label
            start_frame = frame_index

    # Add the last group
    if current_label is not None:
        end_frame = frame_index
        start_time = start_frame / fps
        end_time = end_frame / fps
        grouped_results.append({
            'label': current_label,
            'start_time': start_time,
            'end_time': end_time
        })

    return grouped_results

def count_label_occurrences(predictions):
    """
    Count the occurrences of each label in the predictions.

    :param predictions: Dictionary where keys are frame indices and values are labels.
    :return: Dictionary with labels as keys and their occurrence counts as values.
    """
    # Count the occurrences of each label
    label_counts = Counter(predictions.values())
    
    return label_counts

def get_sample_frame_for_labels(predictions, frames_dir, sample_size=1):
    """
    Get sample frames for each label from the predictions.

    :param predictions: Dictionary where keys are frame indices and values are labels.
    :param frames_dir: Directory where frames are stored.
    :param sample_size: Number of sample frames to extract per label.
    :return: Dictionary where keys are labels and values are frame filenames.
    """
    label_to_frames = {}
    for frame_index, label in predictions.items():
        if label not in label_to_frames:
            label_to_frames[label] = []
        if len(label_to_frames[label]) < sample_size:
            frame_filename = f"frame_{frame_index:04d}.jpg"
            label_to_frames[label].append(frame_filename)
            if len(label_to_frames[label]) >= sample_size:
                if len(label_to_frames) == len(set(predictions.values())):
                    break

    # Choose the first frame for each label
    sample_images_mapped = {label: frames[0] for label, frames in label_to_frames.items()}
    
    return sample_images_mapped

def display_images_from_frames(sample_images_mapped, frames_dir):
    """
    Display images from frames for each label.

    :param sample_images_mapped: Dictionary where keys are labels and values are frame filenames.
    :param frames_dir: Directory where frames are stored.
    """
    for label, frame_file in sample_images_mapped.items():
        frame_path = os.path.join(frames_dir, frame_file)
        image = Image.open(frame_path)
        # Fetch the description for the label
        description = class_description.get(label, "No description available for this label.")

        # Display image with its label
        plt.figure()
        plt.imshow(image)
        plt.title(f"Label: {label}\n Description:{description}")
        plt.axis('off')  # Hide axes

        # Save or show the image
        plt.show()
