
# =====================================================================================
# TASK 2: DATABASE SETUP & CASE PROCESSING
#
# DESCRIPTION:
# This script defines the database structure and includes logic to populate it
# by scanning a configured directory for case data.
# All configuration is loaded from the central 'config.yaml' file.
# =====================================================================================

import os
import json
import yaml  
from typing import Dict, Any, List, TypedDict
from pathlib import Path
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# --- 1. CONFIGURATION LOADING ---

#  Load all settings from the config.yaml file ---
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


# --- 2. DATABASE CONFIGURATION ---

# Get database file name from the loaded config
DATABASE_FILE = CONFIG['database']['file_name']
# The 'engine' is the entry point to our database.
engine = create_engine(f'sqlite:///{DATABASE_FILE}')
# This is the base class our ORM models will inherit from
Base = declarative_base()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# --- 3. DATABASE TABLE DEFINITIONS (ORM MODELS) ---

class Case(Base):
    """
    Represents a single unique case/procedure.
    Corresponds to one folder in the `cases/` directory.
    """
    __tablename__ = 'cases'
    
    id = Column(Integer, primary_key=True)
    case_uid = Column(String, unique=True, nullable=False)
    creation_timestamp = Column(DateTime)
    video_path = Column(String)
    status = Column(String, default="processed")

    # Relationships to link with other tables
    annotations = relationship("FrameAnnotation", back_populates="case", cascade="all, delete-orphan")
    summary = relationship("CaseSummary", back_populates="case", uselist=False, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Case(case_uid='{self.case_uid}')>"

class FrameAnnotation(Base):
    """
    Represents a single detected polyp object in a single frame.
    """
    __tablename__ = 'frame_annotations'
    
    id = Column(Integer, primary_key=True)
    case_id = Column(Integer, ForeignKey('cases.id'), nullable=False)
    
    frame_number = Column(Integer, nullable=False)
    timestamp_ms = Column(Integer)
    object_id = Column(Integer, nullable=False)
    
    class_name = Column(String, nullable=False)
    confidence = Column(Float)
    
    bb_x_min = Column(Integer)
    bb_y_min = Column(Integer)
    bb_x_max = Column(Integer)
    bb_y_max = Column(Integer)

    case = relationship("Case", back_populates="annotations")

    def __repr__(self):
        return f"<Annotation(case_id={self.case_id}, frame={self.frame_number}, object_id={self.object_id})>"

class CaseSummary(Base):
    """
    Stores the AI-generated summary for a case.
    """
    __tablename__ = 'case_summaries'
    
    id = Column(Integer, primary_key=True)
    case_id = Column(Integer, ForeignKey('cases.id'), unique=True, nullable=False)
    
    generated_at = Column(DateTime, default=datetime.utcnow)
    summary_text = Column(Text, nullable=False)
    
    case = relationship("Case", back_populates="summary")

class SummarizerState(TypedDict):
    case_uid: str
    overview: Dict[str, Any]
    polyp_details: List[Dict[str, Any]]
    final_summary: str
    prompt_style: str 


# --- 4. CORE LOGIC FUNCTIONS ---

def create_database_and_tables():
    """
    Creates the SQLite database file and all defined tables if they don't already exist.
    """
    print("--- Initializing Database ---")
    if not os.path.exists(DATABASE_FILE):
        print(f"Database file not found. Creating '{DATABASE_FILE}'...")
    else:
        print(f"Database '{DATABASE_FILE}' already exists. Verifying tables...")
    
    # This command creates tables that are missing, but does not alter existing ones.
    Base.metadata.create_all(engine)
    print("Database and tables are ready.\n")

# The process_and_insert_case function does not need changes, as it receives the path.
def process_and_insert_case(case_folder_path: Path, SessionLocal):
    """
    Reads a single case's JSON file and inserts its data into the database.
    Skips the case if it has already been processed.
    """
    json_path = case_folder_path / "annotations.json"
    if not json_path.exists():
        print(f"  [SKIP] 'annotations.json' not found in '{case_folder_path.name}'.")
        return

    # Check if this case (by folder name) is already in the database
    session = SessionLocal()
    try:
        case_uid = case_folder_path.name
        existing_case = session.query(Case).filter_by(case_uid=case_uid).first()
        if existing_case:
            print(f"  [INFO] Case '{case_uid}' already exists. Skipping.")
            return

        print(f"  [PROCESS] New case found: '{case_uid}'. Reading data...")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Insert the main Case record
        metadata = data.get("case_metadata", {})
        creation_time = datetime.fromisoformat(metadata.get("creation_timestamp_utc")) if metadata.get("creation_timestamp_utc") else None

        new_case = Case(
            case_uid=metadata.get("case_id", case_uid),
            creation_timestamp=creation_time,
            video_path=str(case_folder_path / metadata.get("video_filename", "video.mp4"))
        )
        session.add(new_case)
        # Commit the case to get its auto-generated primary key (new_case.id)
        session.commit()

        # Prepare all frame annotations for bulk insertion
        annotations_to_add = []
        for frame_data in data.get("frame_annotations", []):
            for detection in frame_data.get("detections", []):
                bbox = detection.get("bounding_box_xyxy", [0, 0, 0, 0])
                annotations_to_add.append(FrameAnnotation(
                    case_id=new_case.id, # Link to the parent Case record
                    frame_number=frame_data.get("frame_number"),
                    timestamp_ms=frame_data.get("timestamp_ms"),
                    object_id=detection.get("object_id"),
                    class_name=detection.get("class_name"),
                    confidence=detection.get("confidence"),
                    bb_x_min=bbox[0],
                    bb_y_min=bbox[1],
                    bb_x_max=bbox[2],
                    bb_y_max=bbox[3]
                ))
                
        if annotations_to_add:
            session.bulk_save_objects(annotations_to_add)
            print(f"  [INSERT] Added {len(annotations_to_add)} annotation records.")
        
        # Commit all annotations for this case
        session.commit()
        print(f"  [SUCCESS] Finished processing and saved '{case_uid}'.")
    finally:
        session.close()


def run_processor():
    """
    Main function to scan the 'cases' directory and process each subfolder.
    """
    #  Get the cases directory from the loaded config
    cases_dir = Path(CONFIG['paths']['cases_root_dir'])
    if not cases_dir.exists():
        print(f"[ERROR] '{cases_dir}' directory not found. Please run the data capture script first.")
        return
        
    print("--- Starting Case Processing Scan ---")
    
    # Use the SessionLocal factory directly to ensure a fresh session per case
    for case_folder in cases_dir.iterdir():
        if case_folder.is_dir():
            process_and_insert_case(case_folder, SessionLocal)
            
    print("\n--- Case Processing Scan Finished ---")

# --- 5. SCRIPT EXECUTION ---

if __name__ == "__main__":
    # Step 1: Ensure the database and tables exist
    create_database_and_tables()
    
    # Step 2: Run the processor to find and insert new cases
    run_processor()