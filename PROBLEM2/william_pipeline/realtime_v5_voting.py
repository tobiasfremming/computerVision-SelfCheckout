# Using YOLO for object detection, including detecting multiple objects in scene.
# Can use EfficientNet0 to classfiy more accurately in a sub-thread.

# Requires
# sudo apt update
# sudo apt install -y libgl1

import cv2
import os
import numpy as np
import torch  # Add this import
from ultralytics import YOLO
from sort import Sort
from datetime import timedelta
import pandas as pd
from collections import defaultdict

CONFIDENCE = 0.4  # Confidence threshold for detections
COOLDOWN = 0.5

class_id_to_name = {
    0: "Leverpostei (7037203626563)",
    1: "Epler Røde (4015)",
    2: "Yoghurt Skogsbær (7038010009457)",
    3: "Chips Havsalt (7071688004713)",
    4: "Red Bull SF (90433924)",
    5: "Banan Øko (94011)",
    6: "Appelsin (4196)",
    7: "Skinke (7037206100022)",
    8: "Kaffe Evergood (7040913336684)",
    9: "YT Vanilje (7038010068980)",
    10: "Q Yoghurt (7048840205868)",
    11: "Norvegia (7038010013966)",
    12: "Jarlsberg (7038010021145)",
    13: "Egg 12pk (7039610000318)",
    14: "Paprika (4088)",
    15: "Grove Rundstykker (7035620058776)",
    16: "Pepsi Max (7044610874661)",
    17: "Kvikk Lunsj (7622210410337)",
    18: "Red Bull Reg (90433917)",
    19: "Cottage Cheese (7038010054488)",
    20: "Ruccula (7023026089401)",
    21: "Karbonadedeig (7020097009819)",
    22: "Gulrot 1kg (7040513001753)",
    23: "Gulrot 750g (7040513000022)",
    24: "Kjøttdeig Angus (7020097026113)",
    25: "Banan (4011)"
}

# Define the scan region for detecting 'scan events'
def get_scan_zone(resolution):
    if resolution == 480:
        return (300, 200, 600, 550)
    elif resolution == 720:
        return (450, 320, 820, 740)
    elif resolution == 1080:
        return (700, 350, 1280, 1120)
    else:
        return (300, 400, 600, 600)  # default fallback

def is_inside_scan_area(bbox, scan_zone):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return (scan_zone[0] <= cx <= scan_zone[2] and
            scan_zone[1] <= cy <= scan_zone[3])

def is_leftward_entry(current_pos, history, min_distance=50):
    """Check if an object entered from the left side and moved rightward"""
    if len(history) < 3:  # Need some history to determine movement
        return True  # Default to true (safer)
    
    # Calculate average x-movement over last few positions
    x_movements = [history[i+1][0] - history[i][0] for i in range(len(history)-1)]
    avg_movement = sum(x_movements) / len(x_movements)
    
    # Check if object entered from left (low x) and generally moving rightward
    return history[0][0] < 200 and avg_movement > 0

def process_video(video_path, model, resolution, device='cpu'):
    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.1)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_num = 0
    scan_zone = get_scan_zone(resolution)

    # Add at the beginning of process_video function (after initializing 'cap')
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    scan_threshold_x = int(frame_width * 0.45)  # 45% of frame width (changed from 0.6)
    max_x_positions = {}  # track_id -> max x position reached

    # Track position history for each tracking ID
    position_history = defaultdict(list)  # track_id -> list of (x, y) center positions
    history_limit = 10  # Keep only the last 10 positions
    min_positions_required = 3  # Require at least 3 positions before scanning
    
    # NEW: Track class predictions for voting
    class_votes = defaultdict(lambda: defaultdict(int))  # track_id -> {class_id -> count}
    
    scanned_log = {}  # track_id -> (product_name, timestamp)
    
    # Track recently disappeared objects
    disappeared_objects = {}  
    reappearance_window = int(1 * fps)  # 1 second window for reappearance
    
    # Track instances per class
    instance_counters = {}  # cls_id -> count
    
    # For cooldown between scans
    last_class_scan_time = {}  # class_id -> frame_num
    cooldown_frames = int(COOLDOWN * fps)  # 0.5 seconds worth of frames
    
    # NEW: Add wait period for objects that leave tracking
    pending_scans = {}  # track_id -> [most_voted_cls, vote_count, timestamp, expiry_frame]
    wait_frames = int(COOLDOWN * fps)  # Use COOLDOWN seconds wait before finalizing scan
    
    # NEW: Track objects in scan zone
    objects_in_scan_zone = {}  # track_id -> first_seen_frame
    scan_zone_timeout = int(2 * fps)  # 2 seconds max in scan zone
    
    # NEW: Add a tracking structure for products being processed to avoid duplicate scans
    products_being_processed = {}  # cls_id -> expiry_frame
    
    output_df = []

    print(f"\n🔍 Processing {video_path} at {resolution}p...")
    print(f"Cooldown frames: {cooldown_frames}")
    
    # Create window for display
    cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
    
    # Track which objects were seen in current frame - keyed by (cls_id, instance_id)
    active_objects = set()
    active_tracks = set()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        timestamp = str(timedelta(seconds=frame_num / fps))
        active_objects.clear()
        active_tracks.clear()

        # Make a copy of the frame for drawing
        display_frame = frame.copy()
        
        # Draw scan zone
        cv2.rectangle(display_frame, 
                     (scan_zone[0], scan_zone[1]), 
                     (scan_zone[2], scan_zone[3]), 
                     (0, 255, 0), 2)

        # Draw scan threshold line
        cv2.line(display_frame, 
                 (scan_threshold_x, 0), 
                 (scan_threshold_x, display_frame.shape[0]), 
                 (255, 255, 0), 2)

        # Run detection only every few frames
        if frame_num % 1 == 0:
            results = model(frame, verbose=False, device=device)
            preds = results[0].boxes

            detections = []
            for det in preds:
                conf = det.conf.item()
                cls_id = int(det.cls.item())
                x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
                if conf > CONFIDENCE:
                    detections.append([x1, y1, x2, y2, conf, cls_id])

            dets_np = np.array(detections)
            if len(dets_np) == 0:
                dets_np = np.empty((0, 6))

            tracked_objects = tracker.update(dets_np)
            
            # Process objects with tracking info
            for trk in tracked_objects:
                x1, y1, x2, y2, track_id = map(int, trk[:5])
                active_tracks.add(track_id)
                
                # Calculate center position
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Update position history
                position_history[track_id].append((center_x, center_y))
                if len(position_history[track_id]) > history_limit:
                    position_history[track_id].pop(0)  # Remove oldest position
                
                cls_id = None
                for det in detections:
                    if all(abs(det[i] - trk[i]) < 8 for i in range(4)):  # fuzzy bbox match
                        cls_id = int(det[5])
                        break
                
                # Skip if no class ID was found
                if cls_id is None:
                    continue

                # NEW: Add vote for this class
                class_votes[track_id][cls_id] += 1
                
                # Get or create instance ID for this track
                if track_id not in scanned_log:
                    # Try to match with a disappeared object
                    matched_key = None
                    matched_distance = float('inf')
                    
                    # Look for close matches in recently disappeared objects
                    for key, info in list(disappeared_objects.items()):
                        if key[0] == cls_id and info['track_id'] != track_id:
                            # Check if it was seen recently
                            if frame_num - info['last_seen'] <= reappearance_window:
                                # Calculate position similarity
                                old_bbox = info['bbox']
                                old_center = ((old_bbox[0] + old_bbox[2]) // 2, (old_bbox[1] + old_bbox[3]) // 2)
                                new_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                                
                                # Calculate distance between centers
                                distance = ((old_center[0] - new_center[0]) ** 2 + 
                                           (old_center[1] - new_center[1]) ** 2) ** 0.5
                                
                                # If this is closer than previous matches and was scanned
                                if distance < matched_distance and info['track_id'] in scanned_log:
                                    matched_key = key
                                    matched_distance = distance
                    
                    # If we found a good match and it's within a reasonable distance
                    if matched_key is not None and matched_distance < 200:  # Threshold for considering it same object
                        old_track_id = disappeared_objects[matched_key]['track_id']
                        if old_track_id in scanned_log:
                            # Copy scan info from old track_id to new track_id
                            scanned_log[track_id] = scanned_log[old_track_id]
                            # Show we're reusing a previous scan
                            cv2.putText(display_frame, "REUSED SCAN", (x1, y1-30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                # Assign instance ID
                if cls_id not in instance_counters:
                    instance_counters[cls_id] = 0
                instance_id = instance_counters[cls_id]
                instance_counters[cls_id] += 1
                
                # Mark this object as active
                obj_key = (cls_id, instance_id)
                active_objects.add(obj_key)
                
                # Update disappeared_objects with current position
                disappeared_objects[obj_key] = {
                    'last_seen': frame_num,
                    'bbox': [x1, y1, x2, y2],
                    'track_id': track_id
                }

                # Check if object is in scan zone
                is_in_scan = is_inside_scan_area((x1, y1, x2, y2), scan_zone)
                
                # NEW: Track when objects enter scan zone
                if is_in_scan and track_id not in objects_in_scan_zone and track_id not in scanned_log:
                    objects_in_scan_zone[track_id] = frame_num
                
                # Update max x position for this track
                if track_id not in max_x_positions:
                    max_x_positions[track_id] = center_x
                else:
                    max_x_positions[track_id] = max(max_x_positions[track_id], center_x)
                
                # Draw tracking box
                if is_in_scan:
                    color = (0, 0, 255)  # Red for items in scan zone
                else:
                    color = (255, 0, 0)  # Blue for other tracked items
                
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                
                # Get most voted class if we have votes
                if track_id in class_votes and class_votes[track_id]:
                    most_frequent_cls = max(class_votes[track_id].items(), key=lambda x: x[1])[0]
                    cls_id_to_show = most_frequent_cls
                    vote_count = class_votes[track_id][most_frequent_cls]
                    vote_info = f" ({vote_count} votes)"
                else:
                    cls_id_to_show = cls_id
                    vote_info = ""
                
                # Add tracking ID and class label if available
                product_name = class_id_to_name.get(cls_id_to_show, f"Product_{cls_id_to_show}")
                short_name = product_name.split()[0]
                label = f"ID:{track_id} {short_name}{vote_info}"
                
                cv2.putText(display_frame, label, (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Display current status for objects in scan zone
                if track_id in objects_in_scan_zone:
                    time_in_zone = frame_num - objects_in_scan_zone[track_id]
                    past_cooldown = all(
                        frame_num - last_class_scan_time.get(cls, 0) >= cooldown_frames 
                        for cls in class_votes[track_id].keys()
                    )
                    
                    if track_id not in scanned_log:
                        if not past_cooldown:
                            cv2.putText(display_frame, "COOLDOWN", (x1, y1-30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
                        elif max_x_positions[track_id] < scan_threshold_x:
                            cv2.putText(display_frame, f"WAIT {max_x_positions[track_id]}/{scan_threshold_x}", (x1, y1-30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                        elif time_in_zone < scan_zone_timeout:
                            cv2.putText(display_frame, "COLLECTING VOTES", (x1, y1-30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Check for objects that left the scan zone or timed out
            for track_id in list(objects_in_scan_zone.keys()):
                # Skip if already scanned or still active in frame
                if track_id in scanned_log or track_id in active_tracks:
                    continue
                
                # Object is no longer being tracked - consider it left the frame
                if track_id in class_votes and class_votes[track_id]:
                    # Find the class with the most votes
                    most_voted_cls, vote_count = max(class_votes[track_id].items(), key=lambda x: x[1])
                    
                    # NEW: Check if this product class is already being processed
                    if most_voted_cls in products_being_processed and frame_num < products_being_processed[most_voted_cls]:
                        print(f"⏩ [{timestamp}] Skipping duplicate scan of {class_id_to_name.get(most_voted_cls)} (ID: {track_id})")
                        objects_in_scan_zone.pop(track_id, None)
                        continue
                        
                    # Check if this class has been scanned recently (cooldown)
                    if most_voted_cls not in last_class_scan_time or \
                       frame_num - last_class_scan_time[most_voted_cls] >= cooldown_frames:
                        
                        # Only continue if it passed the threshold and has enough votes
                        if max_x_positions.get(track_id, 0) >= scan_threshold_x and vote_count >= 3:
                            # NEW: Mark this product class as being processed
                            products_being_processed[most_voted_cls] = frame_num + wait_frames
                            
                            # Add to pending scans
                            if track_id not in pending_scans:
                                pending_scans[track_id] = [most_voted_cls, vote_count, timestamp, frame_num + wait_frames]
                                print(f"⌛ [{timestamp}] Waiting {COOLDOWN}s to confirm scan: {class_id_to_name.get(most_voted_cls)} (ID: {track_id})")
                
                # Remove from scan zone tracking
                objects_in_scan_zone.pop(track_id, None)
            
            # NEW: Check pending scans
            for track_id in list(pending_scans.keys()):
                most_voted_cls, vote_count, scan_timestamp, expiry_frame = pending_scans[track_id]
                
                # If the object reappeared in tracking, cancel the pending scan
                if track_id in active_tracks:
                    pending_scans.pop(track_id)
                    print(f"🔄 [{timestamp}] Object reappeared, canceling scan (ID: {track_id})")
                    continue
                    
                # If wait period has elapsed, finalize the scan
                if frame_num >= expiry_frame:
                    product_name = class_id_to_name.get(most_voted_cls, f"Product_{most_voted_cls}")
                    
                    # NEW: Double-check the cooldown before finalizing
                    if most_voted_cls not in last_class_scan_time or \
                       frame_num - last_class_scan_time[most_voted_cls] >= cooldown_frames:
                        
                        scanned_log[track_id] = (product_name, scan_timestamp)
                        last_class_scan_time[most_voted_cls] = frame_num  # Update last scan time
                        output_df.append({
                            "timestamp": scan_timestamp, 
                            "product": product_name, 
                            "track_id": track_id, 
                            "votes": vote_count
                        })
                        print(f"🛒 [{timestamp}] Confirmed scan: {product_name} (ID: {track_id}, Votes: {vote_count})")
                        
                        # NEW: Clean up processed product tracking
                        products_being_processed.pop(most_voted_cls, None)
                    else:
                        print(f"⏭️ [{timestamp}] Cancelled scan due to cooldown: {product_name} (ID: {track_id})")
                        
                    pending_scans.pop(track_id)
            
            # Clean up old disappeared objects that haven't been seen in a while
            for key in list(disappeared_objects.keys()):
                if frame_num - disappeared_objects[key]['last_seen'] > reappearance_window:
                    if key not in active_objects:
                        disappeared_objects.pop(key, None)
                                
        # Add timestamp to the frame
        cv2.putText(display_frame, timestamp, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display the frame
        cv2.imshow("Object Detection", display_frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return pd.DataFrame(output_df)


def main():
    # More detailed CUDA diagnostics
    print("\n----- CUDA Diagnostics -----")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("Apple Silicon MPS is available")
        device = 'mps'  # Apple Silicon
    else:
        print("No GPU acceleration available, using CPU")
        device = 'cpu'
    
    print(f"🚀 Using device: {device}")
    
    # Load model and explicitly move it to the selected device
    model = YOLO("best.pt").to(device)
    
    # Verify model device
    print(f"Model is on device: {next(model.parameters()).device}")
    
    # Rest of your code remains the same
    video_dir = "videos"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]

    for video_file in video_files:
        resolution = int(video_file.split()[-1].replace("P.mp4", ""))
        # Pass the device to process_video
        df = process_video(os.path.join(video_dir, video_file), model, resolution, device)

        # Save results
        base_name = os.path.splitext(video_file)[0]
        csv_path = os.path.join(output_dir, f"{base_name}_receipt.csv")
        df.to_csv(csv_path, index=False)
        print(f"✅ Receipt saved to {csv_path}")

if __name__ == "__main__":
    main()