# Using YOLO for object detection, including detecting multiple objects in scene.
# Can use EfficientNet0 to classfiy more accurately in a sub-thread.

# Requires
# sudo apt update
# sudo apt install -y libgl1

import cv2
import os
import numpy as np
from ultralytics import YOLO
from sort import Sort
from datetime import timedelta
import pandas as pd

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
def get_scan_zone(cap):
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Your chosen normalized coordinates [0..1]
    x1_uv, y1_uv, x2_uv, y2_uv = (0.35, 0.35, 0.7, 1.0)

    x1 = int(x1_uv * width)
    y1 = int(y1_uv * height)
    x2 = int(x2_uv * width)
    y2 = int(y2_uv * height)

    return (x1, y1, x2, y2)



def is_inside_scan_area(bbox, scan_zone):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return (scan_zone[0] <= cx <= scan_zone[2] and
            scan_zone[1] <= cy <= scan_zone[3])
def process_video(video_path, model, resolution):
    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_num = 0
    scan_zone = get_scan_zone(cap)

    scanned_log = {}  # track_id -> (product_name, timestamp)
    
    # Track recently disappeared objects - NEW STRUCTURE
    # Key is (cls_id, instance_id) to allow multiple objects of same class
    disappeared_objects = {}  
    reappearance_window = int(1 * fps)  # Increased to 3 seconds for better recall
    
    # Track instances per class
    instance_counters = {}  # cls_id -> count
    
    # For cooldown between scans
    last_class_scan_time = {}  # class_id -> frame_num
    cooldown_frames = int(0.5 * fps)  # 0.5 seconds worth of frames
    print(f"Cooldown frames: {cooldown_frames}")

    output_df = []

    print(f"\n🔍 Processing {video_path} at {resolution}p...")
    
    # Create window for display
    cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
    
    # Track which objects were seen in current frame - keyed by (cls_id, instance_id)
    active_objects = set()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        timestamp = str(timedelta(seconds=frame_num / fps))
        active_objects.clear()

        # Make a copy of the frame for drawing
        display_frame = frame.copy()
        
        # Draw scan zone
        cv2.rectangle(display_frame, 
                     (scan_zone[0], scan_zone[1]), 
                     (scan_zone[2], scan_zone[3]), 
                     (0, 255, 0), 2)

        # Run detection only every few frames
        if frame_num % 2 == 0:
            results = model(frame, verbose=False)
            preds = results[0].boxes

            detections = []
            for det in preds:
                conf = det.conf.item()
                cls_id = int(det.cls.item())
                x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
                if conf > 0.5:
                    detections.append([x1, y1, x2, y2, conf, cls_id])

            dets_np = np.array(detections)
            if len(dets_np) == 0:
                dets_np = np.empty((0, 6))

            tracked_objects = tracker.update(dets_np)
            
            # Process objects with tracking info
            for trk in tracked_objects:
                x1, y1, x2, y2, track_id = map(int, trk[:5])
                cls_id = None
                for det in detections:
                    if all(abs(det[i] - trk[i]) < 8 for i in range(4)):  # fuzzy bbox match
                        cls_id = int(det[5])
                        break
                
                # Skip if no class ID was found
                if cls_id is None:
                    continue

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

                # Draw tracking box
                if is_inside_scan_area((x1, y1, x2, y2), scan_zone):
                    color = (0, 0, 255)  # Red for items in scan zone
                else:
                    color = (255, 0, 0)  # Blue for other tracked items
                
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                
                # Add tracking ID and class label if available
                product_name = class_id_to_name.get(cls_id, f"Product_{cls_id}")
                short_name = product_name.split()[0]
                label = f"ID:{track_id} {short_name}"
                
                cv2.putText(display_frame, label, (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Process scan events if not already scanned
                if track_id not in scanned_log and is_inside_scan_area((x1, y1, x2, y2), scan_zone):
                    # Check if this class has been scanned recently (cooldown)
                    can_scan = True
                    if cls_id in last_class_scan_time:
                        time_since_last_scan = frame_num - last_class_scan_time[cls_id]
                        if time_since_last_scan < cooldown_frames:
                            can_scan = False
                            cv2.putText(display_frame, "COOLDOWN", (x1, y1-30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
                    
                    if can_scan:
                        scanned_log[track_id] = (product_name, timestamp)
                        last_class_scan_time[cls_id] = frame_num  # Update last scan time
                        output_df.append({"timestamp": timestamp, "product": product_name, "track_id": track_id})
                        print(f"🛒 [{timestamp}] Scanned: {product_name} (ID: {track_id})")
                        cv2.putText(display_frame, "SCANNED", (x1, y1-30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
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
    model = YOLO("best.pt")

    video_dir = "videos"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]

    for video_file in video_files:
        if(video_file == "Nesten alle varer sakte tempo 480P.mp4"):
            continue

        resolution = int(video_file.split()[-1].replace("P.mp4", ""))
        df = process_video(os.path.join(video_dir, video_file), model, resolution)

        # Save results
        base_name = os.path.splitext(video_file)[0]
        csv_path = os.path.join(output_dir, f"{base_name}_receipt.csv")
        df.to_csv(csv_path, index=False)
        print(f"✅ Receipt saved to {csv_path}")

if __name__ == "__main__":
    main()