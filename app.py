from flask import Flask, render_template, request, redirect, url_for
import os
from utils import extract_frames, predict_frame, group_consecutive_frames, count_label_occurrences
from model import load_combined_model
from torchvision import transforms
import cv2
from collections import defaultdict
from celery import Celery

app = Flask(__name__, template_folder='templates')  # Adjust path if needed
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'  # Update this if using a cloud service
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'  # Update this if using a cloud service
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

# Path for uploaded videos and extracted frames
UPLOAD_FOLDER = 'uploads/'
FRAMES_FOLDER = 'static/frames/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAMES_FOLDER, exist_ok=True)

# Define the transformation for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust size if necessary
    transforms.ToTensor(),
])

# Load the model
model = None
def get_model():
    global model
    if model is None:
        print("Loading the model...")
        model = load_combined_model(num_classes=23)  # Lazy load the model
        model.eval()  # Set the model to evaluation mode
    return model
@celery.task
def process_video(video_path):
    # Extract frames from the video
    extract_frames(video_path, FRAMES_FOLDER)

    # Perform frame analysis
    predictions = {}
    fps = extract_fps(video_path)
    if fps is None:
        return "Error: Could not extract FPS from the video", 500
    model = get_model()

    # To store the first frame for each label
    example_images = {}

    # Perform prediction on each frame
    for frame_file in sorted(os.listdir(FRAMES_FOLDER)):
        if frame_file.endswith('.jpg'):
            frame_path = os.path.join(FRAMES_FOLDER, frame_file)
            frame_index = int(os.path.splitext(frame_file)[0].split('_')[-1])
            try:
                predicted_label = predict_frame(model, frame_path, transform)
                predictions[frame_index] = predicted_label

                # Save an example image for each label
                if predicted_label not in example_images:
                    example_images[predicted_label] = frame_path
            except Exception as e:
                print(f"Error processing {frame_file}: {e}")

    # Group consecutive frames with the same predicted label
    aggregated_results = group_consecutive_frames(predictions, fps)

    # Count label occurrences
    label_occurrences = count_label_occurrences(predictions)
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

    # Update image paths for rendering
    images_with_labels = {label: os.path.basename(path) for label, path in example_images.items()}
    return aggregated_results, label_occurrences, images_with_labels
    
@app.route('/test')
def test():
    return render_template('test.html')  # Ensure test.html is in the templates folder

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'video' not in request.files:
            return redirect(request.url)
        
        file = request.files['video']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(video_path)
             
            
            task = process_video.delay(video_path)
            return redirect(url_for('task_status', task_id=task.id))  # Redirect to a status page

    return render_template('index.html')
@app.route('/task-status/<task_id>')
def task_status(task_id):
    task = process_video.AsyncResult(task_id)
    if task.state == 'PENDING':
        # Task is still processing
        return "Task is processing..."
    elif task.state != 'FAILURE':
        # Task completed successfully
        results = task.result
        return render_template('results.html', predictions=results[0], 
                               label_occurrences=results[1], 
                               example_images=results[2])
    else:
        return "Task failed!"
def extract_fps(video_path):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        return None
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return fps

if __name__ == '__main__':
    app.run(debug=True)
