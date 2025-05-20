
"""
DeepSORT (Deep Simple Online and Realtime Tracking): 
    An extension of the SORT algorithm that incorporates a deep appearance descriptor to improve tracking accuracy, especially for objects that are temporarily occluded or leave the frame.
Kalman Filter: 
    Used in DeepSORT to predict object positions and handle noise in detections.
Hungarian Algorithm: 
    Used for data association, matching detected objects to existing tracks.
"""
import cv2
from ultralytics import YOLO    
from deep_sort_realtime.deepsort_tracker import DeepSort 
import datetime

# --- Configuration ---
VIDEO_PATH = "../data/pedestrian.car.mp4"  # Replace with your video file path
OUTPUT_VIDEO_PATH = "output_tracked_video.mp4"
YOLO_MODEL = "yolov8n.pt"  # You can use yolov8s.pt, yolov8m.pt, yolov8l.pt for larger models
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
TRACK_CLASSES = [0, 2, 3, 5, 7] # 0: person, 2: car, 3: motorcycle, 5: bus, 7: truck (COCO dataset classes)
MAX_AGE = 50 # Maximum number of frames a track can be lost before it is deleted
MIN_HITS = 3 # Minimum number of consecutive detections to establish a track

# --- Colors for bounding boxes ---
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)

model = YOLO(YOLO_MODEL) 

# --- Initialize DeepSORT tracker ---
tracker = DeepSort(max_age=MAX_AGE, n_init=MIN_HITS)

cap = cv2.VideoCapture(VIDEO_PATH)  
if not cap.isOpened():
    print(f"Error: Could not open video file {VIDEO_PATH}")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

frame_count = 0
start_time = datetime.datetime.now()

#output_fps = 30  # Output video frames per second
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    # Get original frame dimensions
  
    # Object Detection  
    results = model(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)[0]

    # Prepare detections for DeepSORT
    # DeepSORT expects detections in the format:
    # [ [xmin, ymin, xmax, ymax, confidence, class_id], ... ]
    detections = []
    for result in results:
        for box in result.boxes.xyxy:  # Bounding box coordinates (in resized frame)
            x1, y1, x2, y2 = map(int, box)

        
            detections.append(([x1, y1, x2-x1, y2-y1], 0.9, 'object'))  # Use original-scale coords
            #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Update Tracker
    # # DeepSORT expects [x, y, w, h] for bounding boxes
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw bounding boxes and track IDs
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb() # Get bounding box in (left, top, right, bottom) format
        
        x_min, y_min, x_max, y_max = map(int, ltrb)
        
        # Get the class name (optional, assuming COCO classes)
        class_name = model.names[track.det_class] if hasattr(track, 'det_class') else "Unknown"

        # Draw bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), GREEN, 2)

        # Draw track ID and class name
        text = f"ID: {track_id} {class_name}"
        cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)

    # Calculate and display FPS
    end_time = datetime.datetime.now()
    total_seconds = (end_time - start_time).total_seconds()
    if total_seconds > 0:
        fps_display = f"FPS: {frame_count / total_seconds:.2f}"
    else:
        fps_display = "FPS: Calculating..."
    
    cv2.putText(frame, fps_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
