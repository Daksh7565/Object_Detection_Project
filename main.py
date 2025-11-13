import os
import argparse
import cv2
import numpy as np
import onnxruntime
import time
import uuid
import json
import yaml  
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv
from tracking.Centroid import CentroidTracker

# --- CONFIGURATION ---
load_dotenv()

def load_config(config_path="config.yaml"):
    """Loads the YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            config['model']['class_names'] = {int(k): v for k, v in config['model']['class_names'].items()}
            config['model']['class_colors'] = {int(k): tuple(v) for k, v in config['model']['class_colors'].items()}
            return config
    except FileNotFoundError:
        print(f"[FATAL ERROR] 'config.yaml' not found. Please ensure it is in the root directory.")
        exit()
    except Exception as e:
        print(f"[FATAL ERROR] Error parsing 'config.yaml': {e}")
        exit()

CONFIG = load_config()

def onnx_session(onnx_model_path: str):
    """Creates and configures the ONNX runtime session."""
    if not os.path.exists(onnx_model_path):
        print(f"\n[ERROR] ONNX model not found at: {onnx_model_path}")
        exit()
    
    print(f"Loading ONNX model from: {onnx_model_path}")
    providers = CONFIG['model']['onnx_providers']
    session = onnxruntime.InferenceSession(str(onnx_model_path), providers=providers)
    print(f"ONNX session created using: {session.get_providers()[0]}")
    return session

def preprocess_rgb_optimized(frame, img_size):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_frame = cv2.resize(rgb_frame, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    input_tensor = resized_frame.astype(np.float32)
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    input_tensor = ((input_tensor / 255.0) - MEAN) / STD
    input_tensor = input_tensor.transpose(2, 0, 1)
    return np.expand_dims(input_tensor, axis=0)

def softmax(x, axis=1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def run_detection_pipeline(input_video_path: str, onnx_session, show_live_preview: bool = False):
    """
    Processes a video to detect and track polyps using settings from the global CONFIG.
    """
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Video file not found at: {input_video_path}")

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise IOError(f"Error: Could not open video source '{input_video_path}'.")

    # Get video metadata
    metadata = {
        "source_fps": cap.get(cv2.CAP_PROP_FPS) or 30,
        "original_dimensions": {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        }
    }
    original_width = metadata["original_dimensions"]["width"]
    original_height = metadata["original_dimensions"]["height"]

    #  Get model and tracker settings from CONFIG
    model_config = CONFIG['model']
    tracker_config = CONFIG['tracker']
    img_size = model_config['input_size']
    confidence_threshold = model_config['confidence_threshold']
    class_names = model_config['class_names']
    class_colors = model_config['class_colors']

    # Setup tracker and data structures
    input_name, output_name = onnx_session.get_inputs()[0].name, onnx_session.get_outputs()[0].name
    #  Initialize CentroidTracker with values from config
    ct = CentroidTracker(
        maxDisappeared=tracker_config['max_disappeared'],
        confirm_hits=tracker_config['confirm_hits'],
        history_size=tracker_config['history_size']
    )
    frame_annotations = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # B. PREPROCESS AND RUN INFERENCE
        input_tensor = preprocess_rgb_optimized(frame, img_size)
        raw_output = onnx_session.run([output_name], {input_name: input_tensor})[0]
        probabilities = softmax(raw_output, axis=1)
        pred_mask = np.argmax(probabilities, axis=1).squeeze()
        confidence_map = np.max(probabilities, axis=1).squeeze()
        
        # C. GET RAW DETECTIONS
        current_frame_detections = []
        detected_classes = np.unique(pred_mask)
        for class_id in detected_classes:
            if class_id == 0: continue
            class_mask = (pred_mask == class_id).astype(np.uint8)
            mean_confidence = np.mean(confidence_map[class_mask == 1]) if np.any(class_mask) else 0
            #  Use confidence threshold from config
            if mean_confidence >= confidence_threshold:
                contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    if radius > 5:
                        current_frame_detections.append(((int(x), int(y)), int(radius), int(class_id), mean_confidence))

        # D. UPDATE TRACKER
        confirmed_objects = ct.update(current_frame_detections)

        # E. AGGREGATE ANNOTATIONS FOR THIS FRAME
        if confirmed_objects:
            frame_detections_list = []
            scale_x = original_width / img_size
            scale_y = original_height / img_size

            for objectID, data in confirmed_objects.items():
                orig_center_x = int(data['centroid'][0] * scale_x)
                orig_center_y = int(data['centroid'][1] * scale_y)
                orig_radius = int(data['radius'] * scale_x)
                
                frame_detections_list.append({
                    "object_id": int(objectID),
                    # Use class names from config
                    "class_name": class_names.get(data['class_id'], "Unknown"),
                    "confidence": round(float(data['confidence']), 4),
                    "bounding_box_xyxy": [orig_center_x - orig_radius, orig_center_y - orig_radius, orig_center_x + orig_radius, orig_center_y + orig_radius]
                })
            
            frame_annotations.append({
                "frame_number": frame_count,
                "timestamp_ms": int(cap.get(cv2.CAP_PROP_POS_MSEC)),
                "detections": frame_detections_list
            })

        # F. OPTIONAL VISUALIZATION
        if show_live_preview:
            scale_x, scale_y = original_width / img_size, original_height / img_size
            for objectID, data in confirmed_objects.items():
                orig_center_x = int(data['centroid'][0] * scale_x)
                orig_center_y = int(data['centroid'][1] * scale_y)
                orig_radius = int(data['radius'] * scale_x)
                #  Use class colors and names from config
                color = class_colors.get(data['class_id'], (0, 255, 0))
                label = f"ID {objectID}: {class_names.get(data['class_id'])}"
                cv2.circle(frame, (orig_center_x, orig_center_y), orig_radius, color, 2)
                cv2.putText(frame, label, (orig_center_x - orig_radius, orig_center_y - orig_radius - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow('Live Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_count += 1
    
    cap.release()
    if show_live_preview:
        cv2.destroyAllWindows()
        
    return frame_annotations, metadata

# <<< 2. MAIN EXECUTION BLOCK >>>
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time detection and data capture.")
    #  Removed hardcoded video source, it is now purely a CLI argument
    parser.add_argument("--video", type=str, required=True, help="Path to the video file to process.")
    parser.add_argument("--show", action='store_true', help="Show the live preview window during processing.")
    args = parser.parse_args()

    # --- SETUP ONNX & CASE ---
    print("--- Live Multi-Class Polyp Detection & Data Capture ---")
    #  Get model path from config
    onnx_model_path = CONFIG['paths']['onnx_model']
    session = onnx_session(onnx_model_path)

    case_id = str(uuid.uuid4())
    #  Get cases root directory from config
    cases_root = Path(CONFIG['paths']['cases_root_dir'])
    case_folder = cases_root / case_id
    case_folder.mkdir(parents=True, exist_ok=True)
    json_path = case_folder / "annotations.json"
    
    # Copy source video to case folder
    import shutil
    shutil.copy(args.video, str(case_folder / "video.mp4"))

    print(f"\n--- Starting New Case: {case_id} ---")
    print(f"Processing video: {args.video}")

    # --- CALL THE REUSABLE FUNCTION ---
    annotations, video_meta = run_detection_pipeline(
        input_video_path=args.video,
        onnx_session=session,
        show_live_preview=args.show
    )

    # --- FINALIZE AND SAVE JSON ---
    case_data = {
        "case_metadata": {
            "case_id": case_id,
            "creation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "video_filename": "video.mp4",
            **video_meta
        },
        "model_metadata": {
            "model_path": os.path.basename(onnx_model_path),
            #  Get model metadata from config
            "img_size": CONFIG['model']['input_size'],
            "confidence_threshold": CONFIG['model']['confidence_threshold'],
            "class_names": CONFIG['model']['class_names']
        },
        "frame_annotations": annotations
    }

    with open(json_path, 'w') as f:
        json.dump(case_data, f, indent=4)
    
    print(f"\nSuccessfully saved case {case_id} with {len(annotations)} annotated frames.")
    print("Inference and capture finished.")