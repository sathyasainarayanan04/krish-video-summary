# utils.py

import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Existing code...
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

# Load the CSV file containing labels and descriptions
def load_label_mapping(csv_file_path):
    df = pd.read_csv("label_description.csv")
    return {row['Label']: row['Description'] for _, row in df.iterrows()}

# Load the fine-tuned BERT model and tokenizer
def load_bert_model(model_path, tokenizer_path):
    model = BertForSequenceClassification.from_pretrained("fine_tuned_biobert-20240907T082926Z-001")
    tokenizer = BertTokenizer.from_pretrained("fine_tuned_biobert-20240907T082926Z-001")
    return model, tokenizer

# Get description for a given label
def get_label_description(label, model, tokenizer, label_mapping):
    label_str = index_to_class.get(label, "Unknown Label")
    
    inputs = tokenizer(label_str, return_tensors="pt", truncation=True, padding=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()

    description = label_mapping.get(label_str, "Description not found")
    return description

# Update existing functions if needed
