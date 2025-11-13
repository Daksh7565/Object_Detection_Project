
import os
import argparse
import yaml  
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from sqlalchemy.orm import sessionmaker
from langchain_groq import ChatGroq
from langchain.tools import tool

from langgraph.graph import StateGraph, END

# Import the database engine and table models from our now config-driven Data script
from Data import engine, Case, FrameAnnotation, CaseSummary, SummarizerState

# Load environment variables (for API keys)
load_dotenv()

# --- 1. CONFIGURATION LOADING ---

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



# --- 2. SETUP DATABASE & LLM ---
Session = sessionmaker(bind=engine)

# Get LLM model name from the loaded config
llm = ChatGroq(
    model=CONFIG['agent']['llm_model'],
    groq_api_key=os.getenv("key_sal_groq") # API key remains in .env for security
)


# --- 3. DEFINE THE TOOLS ---
# The tools themselves do not need any changes. They interact with the database
# via the SQLAlchemy session, which is already configured correctly.

@tool
def get_case_statistics(case_uid: str) -> dict:
    """Retrieves key statistics for a given case_uid from the database."""
    print(f"  [Tool] Running get_case_statistics for case: {case_uid}")
    session = Session()
    try:
        case = session.query(Case).filter_by(case_uid=case_uid).first()
        if not case:
            return {"error": f"Case with UID '{case_uid}' not found."}
        annotations = session.query(FrameAnnotation).filter_by(case_id=case.id).all()
        if not annotations:
            return {"case_uid": case_uid, "total_detections": 0, "message": "No polyps detected."}
        
        unique_object_ids = sorted(list({ann.object_id for ann in annotations}))
        polyp_types = {}
        for ann in annotations:
            polyp_types[ann.class_name] = polyp_types.get(ann.class_name, 0) + 1
            
        first_ms = min(ann.timestamp_ms for ann in annotations)
        last_ms = max(ann.timestamp_ms for ann in annotations)

        stats = {
            "case_uid": case_uid,
            "detection_activity_span_seconds": round((last_ms - first_ms) / 1000, 2),
            "total_unique_polyps": len(unique_object_ids),
            "detected_polyp_ids": unique_object_ids,
            "breakdown_by_type": polyp_types,
        }
        return stats
    finally:
        session.close()

@tool
def get_polyp_details(case_uid: str, object_id: int) -> dict:
    """
    Provides a detailed history for a specific polyp (object_id), including
    start time, end time, and total duration of detection.
    """
    print(f"  [Tool] Running get_polyp_details for object_id: {object_id}")
    session = Session()
    try:
        case = session.query(Case).filter_by(case_uid=case_uid).first()
        if not case: return {"error": "Case not found."}
        annotations = session.query(FrameAnnotation).filter(FrameAnnotation.case_id == case.id, FrameAnnotation.object_id == object_id).order_by(FrameAnnotation.timestamp_ms).all()
        if not annotations: return {"error": "Polyp not found."}
        
        avg_conf = sum(ann.confidence for ann in annotations) / len(annotations)
        
        start_time_ms = annotations[0].timestamp_ms
        end_time_ms = annotations[-1].timestamp_ms
        
        start_time_s = round(start_time_ms / 1000, 2)
        end_time_s = round(end_time_ms / 1000, 2)
        duration_s = round(end_time_s - start_time_s, 2)

        details = {
            "object_id": object_id,
            "class_name": annotations[0].class_name,
            "average_confidence": round(avg_conf, 4),
            "start_time_seconds": start_time_s,
            "end_time_seconds": end_time_s,
            "duration_seconds": duration_s,
        }
        return details
    finally:
        session.close()


# --- 4. DEFINE THE GRAPH NODES ---
def get_overview(state: SummarizerState):
    print("--- Step 1: Getting Case Overview ---")
    case_uid = state["case_uid"]
    overview_data = get_case_statistics.invoke({"case_uid": case_uid})
    return {"overview": overview_data}

def get_all_details(state: SummarizerState):
    print("--- Step 2: Getting Details for Each Polyp ---")
    case_uid = state["case_uid"]
    overview = state["overview"]
    polyp_ids = overview.get("detected_polyp_ids", [])
    details_list = [get_polyp_details.invoke({"case_uid": case_uid, "object_id": polyp_id}) for polyp_id in polyp_ids]
    return {"polyp_details": details_list}

def generate_summary(state: SummarizerState):
    print("--- Step 3: Generating Final Summary ---")
    
    # PROMPT 1: NARRATIVE / DEFAULT
    narrative_prompt_template = (
        "You are an expert medical AI assistant. Your task is to generate a structured and insightful summary of a polyp detection procedure based on the provided data.\n\n"
        "## Raw Data for Analysis\n"
        "### Case Overview Data\n"
        "{overview}\n\n"
        "### Individual Polyp Data\n"
        "{details}\n\n"
        "---------------------------------------------------\n\n"
        "## Polyp Detection Procedure Summary\n\n"
        "### Case Overview\n"
        "- **Case UID:** {case_uid}\n"
        "- **Detection Activity Span:** {activity_span} seconds (Time between first and last detection)\n"
        "- **Total Unique Polyps:** {total_polyps}\n"
        "- **Detected Polyp IDs:** {polyp_ids}\n"
        "- **Breakdown by Type:** {breakdown}\n\n"
        "### Final Conclusion\n"
        "After presenting the individual details, write a brief, one-paragraph summary. This conclusion should synthesize the key findings. "
        "Mention the total number of polyps, the most common type found, and any noteworthy observations, such as a polyp with a particularly long duration or high confidence."
    )
    
    # PROMPT 2: TECHNICAL / DATA-DRIVEN
    technical_prompt_template = (
        "You are a clinical data analysis AI. Generate a technical, data-driven summary of the polyp detection procedure. Focus on quantifiable metrics.\n\n"
        "## Raw Data for Analysis\n"
        "### Case Overview Data\n"
        "{overview}\n\n"
        "---------------------------------------------------\n\n"
        "## Technical Procedure Report\n\n"
        "### 1. Case Metrics\n"
        "- **Case UID:** {case_uid}\n"
        "- **Total Unique Polyps Detected:** {total_polyps}\n"
        "- **Detection Activity Span:** {activity_span} seconds\n"
        "- **Polyp Classification Breakdown:** {breakdown}\n\n"
        "### 2. Polyp Manifest\n"
        "For each polyp, provide a dense, one-line summary: Polyp ID [ID] ([Type]): Start=[start]s, End=[end]s, Duration=[duration]s, Avg. Confidence=[confidence].\n\n"
        "### 3. Key Findings (Bulleted List)\n"
        "Synthesize the data into a bulleted list of key findings. Do not use a narrative paragraph. Focus on the most significant data points.\n"
        "- Identify the polyp with the longest detection duration.\n"
        "- Identify the polyp with the highest average confidence score.\n"
        "- State the most frequently detected polyp type and its count.\n"
        "- Mention any other statistically significant observations from the data."
    )
    
    overview = state["overview"]
    polyp_details = state["polyp_details"]
    prompt_style = state.get("prompt_style", "default")

    if prompt_style == "technical":
        print("    [Info] Using TECHNICAL prompt template.")
        prompt_template = technical_prompt_template
        details_str = "\n".join(
            [f"Polyp ID {d['object_id']} ({d['class_name']}): Start={d['start_time_seconds']}s, End={d['end_time_seconds']}s, Duration={d['duration_seconds']}s, Avg. Confidence={d['average_confidence']}"
            for d in polyp_details]
        )
    else:
        print("    [Info] Using NARRATIVE (default) prompt template.")
        prompt_template = narrative_prompt_template
        details_str = "\n".join([f"- {d}" for d in polyp_details])

    prompt = prompt_template.format(
        overview=overview,
        details=details_str,
        case_uid=state['case_uid'],
        activity_span=overview.get('detection_activity_span_seconds', 'N/A'),
        total_polyps=overview.get('total_unique_polyps', 'N/A'),
        polyp_ids=overview.get('detected_polyp_ids', 'N/A'),
        breakdown=overview.get('breakdown_by_type', 'N/A')
    )
    
    response = llm.invoke(prompt)
    return {"final_summary": response.content}

# --- 5. ASSEMBLE THE GRAPH ---
graph = StateGraph(SummarizerState)
graph.add_node("get_overview", get_overview)
graph.add_node("get_all_details", get_all_details)
graph.add_node("generate_summary", generate_summary)
graph.set_entry_point("get_overview")
graph.add_edge("get_overview", "get_all_details")
graph.add_edge("get_all_details", "generate_summary")
graph.add_edge("generate_summary", END)
agent_app = graph.compile()


# --- 6. MAIN EXECUTION FUNCTION ---
def generate_summary_for_case(case_uid: str, prompt_style: str = "default") -> str:
    print(f"--- Starting Structured Summary Generation for Case: {case_uid} (Style: {prompt_style}) ---")
    initial_state = {"case_uid": case_uid, "prompt_style": prompt_style}
    final_state = agent_app.invoke(initial_state)
    summary_text = final_state["final_summary"]

    #  Get the cases directory from the loaded config
    cases_root_dir = Path(CONFIG['paths']['cases_root_dir'])
    case_folder = cases_root_dir / case_uid
    if case_folder.exists():
        summary_path = case_folder / "summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        print(f"\n  [SUCCESS] Summary saved to file: {summary_path}")
    
    session = Session()
    try:
        case = session.query(Case).filter_by(case_uid=case_uid).first()
        if case:
            existing_summary = session.query(CaseSummary).filter_by(case_id=case.id).first()
            if existing_summary:
                existing_summary.summary_text = summary_text
                existing_summary.generated_at = datetime.utcnow()
            else:
                new_summary = CaseSummary(case_id=case.id, summary_text=summary_text)
                session.add(new_summary)
            session.commit()
            print("  [SUCCESS] Summary saved to database.")
    finally:
        session.close()

    return summary_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an AI summary for a polyp detection case.")
    parser.add_argument("--case_uid", required=True, type=str, help="The unique ID of the case to summarize.")
    args = parser.parse_args()
    clean_case_uid = args.case_uid.strip().lstrip(':')
    summary = generate_summary_for_case(clean_case_uid)
    print("\n--- Generated Summary ---")
    print(summary)
    print("-------------------------")