import streamlit as st
import requests
import pandas as pd
from pathlib import Path
from fastapi import UploadFile

# --- CONFIGURATION ---
API_BASE_URL = "http://127.0.0.1:8000"
CASES_ROOT_DIR = Path("cases")

# --- PAGE SETUP ---
st.set_page_config(
    page_title="Polyp Detection AI Assistant",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Polyp Detection AI Case Analysis")
option = ["NEW CASE", "View CASE"]
c = st.sidebar.selectbox("select the option ", option)

# --- API HELPER FUNCTIONS ---

def get_case_details(case_uid: str):
    """Fetches the full details for a single case from the API."""
    try:
        response = requests.get(f"{API_BASE_URL}/cases/{case_uid}/")
        if response.status_code == 404:
            st.error(f"Case with UID '{case_uid}' not found. Please check the ID.")
            return None
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to the API: {e}")
        return None

def generate_summary(case_uid: str, style: str = "default"):
    """Triggers the AI agent to generate a summary for a case with a specific style."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/cases/{case_uid}/summary/generate",
            json={"style": style}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error generating summary: {e}")
        return None

def add_data(video_file: UploadFile) -> dict:
    """Sends the video file to the API to create a new case."""
    try:
        files = {'video_file': (video_file.name, video_file.getvalue(), video_file.type)}
        response = requests.post(f"{API_BASE_URL}/cases/", files=files)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error creating case: {e}")
        return None

def update_summary(case_uid: str, new_text: str):
    """Sends a request to the API to update the summary text."""
    try:
        response = requests.put(
            f"{API_BASE_URL}/cases/{case_uid}/summary/",
            json={"summary_text": new_text}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error updating summary: {e}")
        return None

# --- DATA PROCESSING FUNCTIONS ---

def process_annotations_for_table(annotations: list) -> pd.DataFrame:
    """Creates the top-level summary table using frame numbers and timestamps."""
    if not annotations:
        return pd.DataFrame()
    
    df = pd.DataFrame(annotations)
    
    agg_df = df.groupby('object_id').agg(
        class_name=('class_name', 'first'),
        start_frame=('frame_number', 'min'),
        end_frame=('frame_number', 'max'),
        start_time_ms=('timestamp_ms', 'min'),
        end_time_ms=('timestamp_ms', 'max'),
        avg_confidence=('confidence', 'mean')
    ).reset_index()

    agg_df['start_time_s'] = (agg_df['start_time_ms'] / 1000).round(2)
    agg_df['end_time_s'] = (agg_df['end_time_ms'] / 1000).round(2)
    agg_df['duration_frames'] = agg_df['end_frame'] - agg_df['start_frame']
    agg_df['timeframe'] = agg_df.apply(lambda row: f"Frame {row['start_frame']} - {row['end_frame']}", axis=1)

    final_table = agg_df[[
        'class_name', 'start_time_s', 'end_time_s', 'timeframe', 'duration_frames', 'avg_confidence'
    ]].copy()
    
    final_table.columns = [
        "Class", "Start Time (s)", "End Time (s)", "Timeframe (frames)", "Duration (frames)", "Avg. Confidence"
    ]
    
    final_table['Avg. Confidence'] = final_table['Avg. Confidence'].map('{:.4f}'.format)
    return final_table

# --- MAIN PAGE LAYOUT ---
if c == option[1]: # VIEW CASE
    st.subheader("Search for a Case")

    # --- STATE MANAGEMENT FOR VIEW CASE ---
    if 'view_summary' not in st.session_state:
        st.session_state.view_summary = None

    def clear_view_state():
        """Callback function to clear the summary when the input changes."""
        st.session_state.view_summary = None

    case_uid_input = st.text_input(
        "Enter the Case UID to load its details:",
        placeholder="e.g., 123e4567-e89b-12d3-a456-426614174000",
        key="case_uid_input",
        on_change=clear_view_state # Clear state when user types a new ID
    )

    if case_uid_input:
        details = get_case_details(case_uid_input.strip())

        if details:
            st.divider()
            st.header(f"Analysis for Case: `{details['case_uid']}`")

            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Polyp Details Summary")
                polyp_table = process_annotations_for_table(details.get("annotations", []))
                if not polyp_table.empty:
                    st.dataframe(polyp_table, use_container_width=True)
                else:
                    st.info("This case has no polyp annotations recorded.")
            with col2:
                st.subheader("Procedure Video")
                video_path = CASES_ROOT_DIR / details['case_uid'] / "video.mp4"
                if video_path.exists():
                    st.video(str(video_path))
                else:
                    st.warning("Video file not found on the server.")

            st.divider()
            st.subheader("AI Generated Summary")

            # --- UPDATED DISPLAY LOGIC ---
            # Prioritize showing the newly generated summary from session state.
            # Otherwise, fall back to the summary loaded with the case details.
            summary_to_display = st.session_state.view_summary or details.get("summary")

            if summary_to_display:
                edited_summary = st.text_area(
                    label="Edit Summary",
                    value=summary_to_display["summary_text"],
                    height=300,
                    key=f"summary_editor_{details['case_uid']}"
                )
                
                if st.button("💾 Save Edited Summary", key=f"save_summary_{details['case_uid']}"):
                    with st.spinner("Saving changes..."):
                        updated = update_summary(details['case_uid'], edited_summary)
                        if updated:
                            # Store the updated summary and rerun to confirm
                            st.session_state.view_summary = updated
                            st.success("Summary updated successfully!")
                            st.rerun()
            else:
                st.info("No AI summary available for this case. You can generate one below.")

            st.write("Choose a summary style to generate or re-generate:")
            col1_btn, col2_btn, col_spacer = st.columns([2, 2, 3])

            with col1_btn:
                if st.button("🔄 Generate Standard Summary", key=f"regen_{details['case_uid']}", use_container_width=True):
                    with st.spinner("🤖 The AI agent is writing a narrative summary..."):
                        new_summary = generate_summary(details['case_uid'], style="default")
                        if new_summary:
                            # Store the result in session state BEFORE rerunning
                            st.session_state.view_summary = new_summary
                            st.success("Standard summary generated!")
                            st.write(new_summary)
            
            
            with col2_btn:
                if st.button("📊 Generate Technical Report", key=f"regen_tech_{details['case_uid']}", use_container_width=True):
                    with st.spinner("🤖 The AI agent is compiling a technical report..."):
                        new_summary = generate_summary(details['case_uid'], style="technical")
                        if new_summary:
                            # Store the result in session state BEFORE rerunning
                            st.session_state.view_summary = new_summary
                            st.success("Technical report generated!")
                            st.write(new_summary)
                            
            
else: # NEW CASE
    st.subheader("Create a New Case")
    uploaded_file = st.file_uploader("Upload a video file (MP4 format)", type=["mp4"], key="new_case_uploader")

    if 'new_case_data' not in st.session_state:
        st.session_state.new_case_data = None
    if 'generated_summary' not in st.session_state:
        st.session_state.generated_summary = None

    if uploaded_file is not None:
        st.video(uploaded_file)
        if st.button("Process and Create Case"):
            st.session_state.new_case_data = None
            st.session_state.generated_summary = None
            
            with st.spinner("Uploading and processing video... This may take a moment."):
                case_data = add_data(video_file=uploaded_file)
                if case_data:
                    st.session_state.new_case_data = case_data
                else:
                    st.session_state.new_case_data = None
            st.rerun()

    if st.session_state.new_case_data:
        case_uid = st.session_state.new_case_data.get('case_uid')
        st.success(f"Successfully created a new case with UID: `{case_uid}`")
        st.json(st.session_state.new_case_data)
        
        st.divider()

        if st.button("🤖 Generate AI Summary"):
            if case_uid:
                with st.spinner("The AI agent is analyzing the case... This may take a moment."):
                    summary_response = generate_summary(case_uid, style="default")
                    if summary_response:
                        st.session_state.generated_summary = summary_response
                    else:
                        st.session_state.generated_summary = None
            st.rerun()

    if st.session_state.generated_summary:
        st.success("Summary generated successfully!")
        st.subheader("Generated Summary")
        st.text_area(
            label="Summary Details", 
            value=st.session_state.generated_summary.get('summary_text', 'No summary text found.'), 
            height=300,
            key="new_summary_display"
        )
