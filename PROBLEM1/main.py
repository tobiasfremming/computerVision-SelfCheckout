import os
import json
import glob
import argparse
import numpy as np
import re
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from ultralytics import YOLO

def load_model(model_path):
    """Load the YOLO model using ultralytics."""
    model = YOLO(model_path)
    return model

# Hardcoded mapping from class index to product code
CLASS_MAPPING = {
    0: "7037203626563",  # Leverpostei
    1: "4015",           # Epler Røde
    2: "7038010009457",  # Yoghurt Skogsbær
    3: "7071688004713",  # Chips Havsalt
    4: "90433924",       # Red Bull SF
    5: "94011",          # Banan Øko
    6: "4196",           # Appelsin
    7: "7037206100022",  # Skinke
    8: "7040913336684",  # Kaffe Evergood
    9: "7038010068980",  # YT Vanilje
    10: "7048840205868", # Q Yoghurt
    11: "7038010013966", # Norvegia
    12: "7038010021145", # Jarlsberg
    13: "7039610000318", # Egg 12pk
    14: "4088",          # Paprika
    15: "7035620058776", # Grove Rundstykker
    16: "7044610874661", # Pepsi Max
    17: "7622210410337", # Kvikk Lunsj
    18: "90433917",      # Red Bull Reg
    19: "7038010054488", # Cottage Cheese
    20: "7023026089401", # Ruccula
    21: "7020097009819", # Karbonadedeig
    22: "7040513001753", # Gulrot 1kg
    23: "7040513000022", # Gulrot 750g
    24: "7020097026113", # Kjøttdeig Angus
    25: "4011",          # Banan
}

def load_ground_truth(txt_path):
    """Load ground truth bounding box from a text file."""
    with open(txt_path, 'r') as f:
        data = json.load(f)
    
    labels = []
    for label_info in data['label']:
        labels.append({
            'label': label_info['label'],
            'bbox': [
                label_info['topX'], 
                label_info['topY'], 
                label_info['bottomX'], 
                label_info['bottomY']
            ]
        })
    return labels

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes.
    
    box format: [topX, topY, bottomX, bottomY]
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    # Check if there is no intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # Calculate the area of intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate the Union area by using formula: Union = A + B - Intersection
    union_area = box1_area + box2_area - intersection_area
    
    # Compute the IoU
    iou = intersection_area / union_area
    
    return iou

def process_image(model, image_path, txt_path, iou_threshold=0.5):
    """Process a single image and compare with ground truth."""
    # Load the image
    image = Image.open(image_path)
    
    # Get predictions from the model
    results = model(image)
    
    # Convert results to standard format
    predictions = []
    for detection in results[0].boxes:
        # Get the box coordinates (normalized)
        box = detection.xyxyn[0]  # normalized xmin, ymin, xmax, ymax
        x1, y1, x2, y2 = box
        
        # Get class and confidence
        cls_idx = int(detection.cls[0])
        conf = float(detection.conf[0])
        
        # Get the product code from the class index
        product_code = CLASS_MAPPING.get(cls_idx)
        
        predictions.append({
            'class_idx': cls_idx,
            'product_code': product_code,
            'bbox': [x1, y1, x2, y2],
            'confidence': conf
        })
    
    # Load ground truth
    ground_truth = load_ground_truth(txt_path)
    
    # Match predictions with ground truth using IoU
    matched = []
    for gt in ground_truth:
        best_iou = 0
        best_pred = None
        
        for pred in predictions:
            if pred in matched:
                continue
            
            # Check if the ground truth label matches the product code
            if gt['label'] == pred['product_code']:
                iou = calculate_iou(gt['bbox'], pred['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_pred = pred
        
        if best_iou >= iou_threshold and best_pred is not None:
            matched.append(best_pred)
    
    # Calculate metrics
    true_positives = len(matched)
    false_positives = len(predictions) - true_positives
    false_negatives = len(ground_truth) - true_positives
    
    result = {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'ground_truth_count': len(ground_truth),
        'prediction_count': len(predictions)
    }
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Evaluate YOLO model on product images')
    parser.add_argument('--model', type=str, default='best.pt', help='Path to YOLO model')
    parser.add_argument('--data-dir', type=str, default='NGD_HACK', help='Directory containing image data')
    parser.add_argument('--iou-threshold', type=float, default=0.5, help='IoU threshold for matching predictions')
    args = parser.parse_args()
    
    # Load the model
    print(f"Loading model from {args.model}...")
    model = load_model(args.model)
    
    # Get all image directories
    image_dirs = [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))]
    
    # Initialize metrics
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    # Process each image
    for dir_name in image_dirs:
        dir_path = os.path.join(args.data_dir, dir_name)
        image_files = glob.glob(os.path.join(dir_path, "*.png"))
        
        # Filter out *_bb.png files (they are visualization images with bounding boxes)
        image_files = [f for f in image_files if not f.endswith('_bb.png')]
        
        print(f"Processing {len(image_files)} images in {dir_name}...")
        
        for image_path in tqdm(image_files):
            txt_path = image_path.replace('.png', '.txt')
            
            # Skip if the text file doesn't exist
            if not os.path.exists(txt_path):
                print(f"Warning: No matching text file for {image_path}")
                continue
            
            # Process the image
            result = process_image(model, image_path, txt_path, args.iou_threshold)
            
            # Update metrics
            total_tp += result['true_positives']
            total_fp += result['false_positives']
            total_fn += result['false_negatives']
    
    # Calculate overall metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\nEvaluation Results:")
    print(f"True Positives: {total_tp}")
    print(f"False Positives: {total_fp}")
    print(f"False Negatives: {total_fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"Accuracy: {total_tp / (total_tp + total_fp + total_fn):.4f}")
    print(f"Percent correct: {total_tp / (total_tp + total_fn) * 100:.2f}%")

if __name__ == "__main__":
    main()
