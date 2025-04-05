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
def get_scan_zone(resolution):
    if resolution == 480:
        return (300, 200, 560, 550)
    elif resolution == 720:
        return (260, 440, 720, 640)
    elif resolution == 1080:
        return (330, 380, 950, 650)  # corrected the y2 value
    else:
        return (300, 400, 600, 600)  # default fallback

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
    scan_zone = get_scan_zone(resolution)

    scanned_log = {}  # track_id -> (product_name, timestamp)

    last_class_scan_time = {}  # class_id -> frame_num
    cooldown_frames = int(0.5 * fps)  # 0.3 seconds worth of frames
    print(cooldown_frames)

    output_df = []

    print(f"\n🔍 Processing {video_path} at {resolution}p...")
    
    # Create window for display
    cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        timestamp = str(timedelta(seconds=frame_num / fps))

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

            # Draw detected objects and tracking info
            for trk in tracked_objects:
                x1, y1, x2, y2, track_id = map(int, trk[:5])
                cls_id = None
                for det in detections:
                    if all(abs(det[i] - trk[i]) < 8 for i in range(4)):  # fuzzy bbox match
                        cls_id = int(det[5])
                        break

                # Draw tracking box
                if is_inside_scan_area((x1, y1, x2, y2), scan_zone):
                    color = (0, 0, 255)  # Red for items in scan zone
                else:
                    color = (255, 0, 0)  # Blue for other tracked items
                
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                
                # Add tracking ID and class label if available
                label = f"ID:{track_id}"
                if cls_id is not None:
                    product_name = class_id_to_name.get(cls_id, f"Product_{cls_id}")
                    short_name = product_name.split()[0]  # Use first word to keep label short
                    label += f" {short_name}"
                
                cv2.putText(display_frame, label, (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Process scan events as before
                if cls_id is not None and track_id not in scanned_log:
                    if is_inside_scan_area((x1, y1, x2, y2), scan_zone):
                        # Check if this class has been scanned recently
                        can_scan = True
                        if cls_id in last_class_scan_time:
                            time_since_last_scan = frame_num - last_class_scan_time[cls_id]
                            if time_since_last_scan < cooldown_frames:
                                can_scan = False
                                # Optionally add debug info
                                # print(f"Cooldown active for {class_id_to_name.get(cls_id)} ({time_since_last_scan/fps:.1f}s)")
                        if can_scan:
                            product_name = class_id_to_name.get(cls_id, f"Product_{cls_id}")
                            scanned_log[track_id] = (product_name, timestamp)
                            last_class_scan_time[cls_id] = frame_num  # Update last scan time
                            output_df.append({"timestamp": timestamp, "product": product_name, "track_id": track_id})
                            print(f"🛒 [{timestamp}] Scanned: {product_name} (ID: {track_id})")
                                
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
        resolution = int(video_file.split()[-1].replace("P.mp4", ""))
        df = process_video(os.path.join(video_dir, video_file), model, resolution)

        # Save results
        base_name = os.path.splitext(video_file)[0]
        csv_path = os.path.join(output_dir, f"{base_name}_receipt.csv")
        df.to_csv(csv_path, index=False)
        print(f"✅ Receipt saved to {csv_path}")

if __name__ == "__main__":
    main()