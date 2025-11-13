
import shutil
import os
import uuid
import onnxruntime
import json
import yaml  
from pathlib import Path
from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile
from sqlalchemy.orm import Session, joinedload
from typing import List, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, ConfigDict

# Import your functions and models from the now config-driven scripts
from main import run_detection_pipeline
from Data import SessionLocal, Case, FrameAnnotation, CaseSummary
from agent import generate_summary_for_case

# --- 1. CONFIGURATION LOADING ---

# Load all settings from the config.yaml file ---
def load_config(config_path="config.yaml"):
    """Loads the YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"[FATAL ERROR] 'config.yaml' not found. Please ensure it is in the root directory.")
        exit()
    except Exception as e:
        print(f"[FATAL ERROR] Error parsing 'config.yaml': {e}")
        exit()

CONFIG = load_config()


# --- 2. GLOBAL SETUP (ONNX, FastAPI App) ---

# Get ONNX model path and providers from config
ONNX_MODEL_PATH = CONFIG['paths']['onnx_model']
ONNX_PROVIDERS = CONFIG['model']['onnx_providers']
ONNX_SESSION = None
try:
    print("API Server: Loading ONNX model...")
    ONNX_SESSION = onnxruntime.InferenceSession(str(ONNX_MODEL_PATH), providers=ONNX_PROVIDERS)
    print(f"API Server: ONNX model loaded using {ONNX_SESSION.get_providers()[0]}")
except Exception as e:
    print(f"[FATAL API ERROR] Could not load ONNX model on startup: {e}")

# Get FastAPI metadata from config
app = FastAPI(
    title=CONFIG['api']['title'],
    description=CONFIG['api']['description'],
    version=CONFIG['api']['version']
)



# ... Pydantic Models ...
class CaseUpdateRequest(BaseModel):
    status: Optional[str] = None
class FrameAnnotationResponse(BaseModel):
    frame_number: int; object_id: int; timestamp_ms: int; class_name: str; confidence: float
    model_config = ConfigDict(from_attributes=True)
class CaseSummaryResponse(BaseModel):
    generated_at: datetime; summary_text: str
    model_config = ConfigDict(from_attributes=True)
class SummaryGenerationRequest(BaseModel):
    style: str = "default"
class SummaryUpdateRequest(BaseModel):
    summary_text: str
class CaseBase(BaseModel):
    case_uid: str; status: str; video_path: Optional[str] = None
class CaseDetailsResponse(CaseBase):
    annotations: List[FrameAnnotationResponse] = []; summary: Optional[CaseSummaryResponse] = None
    model_config = ConfigDict(from_attributes=True)
class CaseListResponse(CaseBase):
    total_annotations: int; has_summary: bool
    model_config = ConfigDict(from_attributes=True)
class StandardResponse(BaseModel):
    message: str


def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

@app.get("/", tags=["General"])
def read_root():
    return {"message": "Welcome to the Polyp Detection Case Management API"}


# --- API ENDPOINTS ---

@app.post("/cases/", response_model=CaseDetailsResponse, status_code=status.HTTP_201_CREATED, tags=["Cases (CRUD)"])
def create_case_from_video_upload(db: Session = Depends(get_db), video_file: UploadFile = File(...)):
    if ONNX_SESSION is None:
        raise HTTPException(status_code=503, detail="Model is not available. Server is not ready.")
    if not video_file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video file.")

    case_uid = str(uuid.uuid4())
    #  Get cases directory from config
    case_folder = Path(CONFIG['paths']['cases_root_dir']) / case_uid
    case_folder.mkdir(parents=True, exist_ok=True)
    video_path = case_folder / "video.mp4"

    try:
        with video_path.open("wb") as buffer:
            shutil.copyfileobj(video_file.file, buffer)
    finally:
        video_file.file.close()

    new_case = Case(case_uid=case_uid, video_path=str(video_path.resolve()), status="processing")
    db.add(new_case)
    db.commit()
    db.refresh(new_case)

    try:
        # The pipeline function now uses settings from its own config load, ensuring consistency.
        frame_annotations, video_meta = run_detection_pipeline(
            input_video_path=str(video_path),
            onnx_session=ONNX_SESSION,
            show_live_preview=False
        )
        
        json_path = case_folder / "annotations.json"
        case_data_for_json = {
            "case_metadata": {
                "case_id": case_uid,
                "creation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "video_filename": "video.mp4",
                **video_meta
            },
            "model_metadata": { "model_path": os.path.basename(ONNX_MODEL_PATH) },
            "frame_annotations": frame_annotations
        }
        with open(json_path, 'w') as f:
            json.dump(case_data_for_json, f, indent=4)

        annotations_to_add = []
        for frame_data in frame_annotations:
            for detection in frame_data.get("detections", []):
                bbox = detection.get("bounding_box_xyxy")
                annotations_to_add.append(FrameAnnotation(
                    case_id=new_case.id, frame_number=frame_data["frame_number"],
                    timestamp_ms=frame_data["timestamp_ms"], object_id=detection["object_id"],
                    class_name=detection["class_name"], confidence=detection["confidence"],
                    bb_x_min=bbox[0], bb_y_min=bbox[1], bb_x_max=bbox[2], bb_y_max=bbox[3]
                ))
        
        if annotations_to_add:
            db.bulk_save_objects(annotations_to_add)

        new_case.status = "processed"
        db.commit()
        db.refresh(new_case)

    except Exception as e:
        new_case.status = "failed"
        db.commit()
        print(f"Error processing case {case_uid}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process video: {str(e)}")

    return new_case

# The following endpoints 
# either abstract (database interaction) or correctly uses configured paths.

@app.get("/cases/", response_model=List[CaseListResponse], tags=["Cases (CRUD)"])
def list_cases(db: Session = Depends(get_db)):
    cases = db.query(Case).options(joinedload(Case.annotations), joinedload(Case.summary)).all()
    response = [CaseListResponse(
        case_uid=case.case_uid, status=case.status, video_path=case.video_path,
        total_annotations=len(case.annotations), has_summary=case.summary is not None
    ) for case in cases]
    return response

@app.get("/cases/{case_uid}/", response_model=CaseDetailsResponse, tags=["Cases (CRUD)"])
def get_case_details(case_uid: str, db: Session = Depends(get_db)):
    case = db.query(Case).filter(Case.case_uid == case_uid).options(
        joinedload(Case.annotations), joinedload(Case.summary)
    ).first()
    if not case:
        raise HTTPException(status_code=404, detail=f"Case with UID '{case_uid}' not found.")
    return case

@app.put("/cases/{case_uid}/", response_model=CaseDetailsResponse, tags=["Cases (CRUD)"])
def update_case(case_uid: str, update_data: CaseUpdateRequest, db: Session = Depends(get_db)):
    case = db.query(Case).filter(Case.case_uid == case_uid).first()
    if not case:
        raise HTTPException(status_code=404, detail=f"Case with UID '{case_uid}' not found.")
    update_dict = update_data.model_dump(exclude_unset=True)
    for key, value in update_dict.items():
        setattr(case, key, value)
    db.commit()
    db.refresh(case)
    return case

@app.put("/cases/{case_uid}/summary/", response_model=CaseSummaryResponse, tags=["Cases (CRUD)"])
def update_summary(case_uid: str, summary_data: SummaryUpdateRequest, db: Session = Depends(get_db)):
    case = db.query(Case).options(joinedload(Case.summary)).filter(Case.case_uid == case_uid).first()
    if not case:
        raise HTTPException(status_code=404, detail=f"Case with UID '{case_uid}' not found.")
    if not case.summary:
        raise HTTPException(status_code=404, detail=f"Summary not found for case '{case_uid}'. Please generate one first.")
    case.summary.summary_text = summary_data.summary_text
    db.commit()
    db.refresh(case.summary)
    return case.summary

@app.delete("/cases/{case_uid}/", response_model=StandardResponse, tags=["Cases (CRUD)"])
def delete_case(case_uid: str, db: Session = Depends(get_db)):
    case = db.query(Case).filter(Case.case_uid == case_uid).first()
    if not case:
        raise HTTPException(status_code=404, detail=f"Case with UID '{case_uid}' not found.")
    
    #  Get cases directory from config for robust deletion
    case_folder = Path(CONFIG['paths']['cases_root_dir']) / case_uid
    
    # Perform database deletion first
    db.delete(case)
    db.commit()
    
    # Then delete the folder from the file system
    if case_folder.exists():
        try:
            shutil.rmtree(case_folder)
        except OSError as e:
            # This is not a fatal error for the API request, so just log it.
            print(f"Error removing folder {case_folder}: {e}")
            
    return {"message": f"Case '{case_uid}' and its associated files have been deleted successfully."}

@app.post("/cases/{case_uid}/summary/generate", response_model=CaseSummaryResponse, tags=["Agent"])
def run_agent_to_generate_summary(
    case_uid: str,
    request_data: SummaryGenerationRequest,
    db: Session = Depends(get_db)
):
    """Trigger the AI agent to generate (or re-generate) a summary."""
    case = db.query(Case).filter(Case.case_uid == case_uid).first()
    if not case:
        raise HTTPException(status_code=404, detail=f"Case with UID '{case_uid}' not found.")
    try:
        # The agent function now loads its own config, ensuring consistency.
        generate_summary_for_case(case_uid, prompt_style=request_data.style)
        
        db.refresh(case)
        if not case.summary:
             raise HTTPException(status_code=500, detail="Agent ran but failed to save summary.")
        return case.summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")