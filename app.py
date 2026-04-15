import streamlit as st
import os
import librosa
import numpy as np
import time
from backend.pipeline import SpeakPipeline
from backend.mvp_dataset_loader import MVPDatasetLoader

# This prevents the "503 Service Unavailable" check
os.environ["TRANSFORMERS_OFFLINE"] = "1" 
os.environ["HF_DATASETS_OFFLINE"] = "1"

from backend.pipeline import SpeakPipeline
from backend.mvp_dataset_loader import MVPDatasetLoader

# --- Page Config ---
st.set_page_config(page_title="Speak Prototype", layout="wide")

# --- Terminal Helper ---
def log_to_terminal(message):
    print(f"[SYSTEM LOG] {time.strftime('%H:%M:%S')} - {message}", flush=True)

# --- Initialize Pipeline ---
@st.cache_resource
def load_pipeline():
    log_to_terminal("Starting Load Pipeline...")
    
    # Visual feedback for the user
    progress_bar = st.progress(0, text="Initializing Model Loading...")
    
    log_to_terminal("Checking for GPU/CUDA...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_to_terminal(f"Target Device: {device}")
    progress_bar.progress(20, text=f"Device Found: {device}. Loading Whisper weights...")

    log_to_terminal("Loading SpeakPipeline (Whisper-Medium + LoRA)...")
    pipe = SpeakPipeline(
        device=device,
        whisper_model_name="openai/whisper-medium",
        lora_path="models/checkpoint-1488",
        mapper_model_path="models/artifacts/mapper_model.joblib",
        label_encoder_path="models/artifacts/label_encoder.joblib"
    )
    
    log_to_terminal("Pipeline successfully loaded into memory.")
    progress_bar.progress(100, text="Model Loaded Successfully!")
    time.sleep(1) # Brief pause so user sees the 100%
    progress_bar.empty()
    
    return pipe

@st.cache_resource
def load_dataset():
    log_to_terminal("Accessing Dataset Index...")
    loader = MVPDatasetLoader(dataset_path="dataset/easycall_mvp_dataset")
    loader.load()
    log_to_terminal(f"Dataset Loaded. Found {len(loader.commands)} command classes.")
    return loader

# Initialize
try:
    import torch
    # These will trigger the terminal logs and progress bar
    pipeline = load_pipeline()
    dataset_loader = load_dataset()
except Exception as e:
    log_to_terminal(f"CRITICAL ERROR: {str(e)}")
    st.error(f"Error loading models or dataset: {e}")

# --- UI Layout ---
st.title("🗣️ Speak Prototype")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Audio Input Source")
    input_mode = st.radio("Choose Input:", ["Upload Audio", "easycall-mvp-dataset"], horizontal=True)

    selected_audio = None

    if input_mode == "Upload Audio":
        uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])
        if uploaded_file:
            log_to_terminal(f"User uploaded file: {uploaded_file.name}")
            audio_bytes, _ = librosa.load(uploaded_file, sr=16000)
            selected_audio = audio_bytes
            st.audio(uploaded_file)

    else:
        commands = dataset_loader.commands
        target_cmd = st.selectbox("Select Command Class:", commands)
        
        samples = dataset_loader.dataset.filter(lambda x: x['text'] == target_cmd)
        sample_indices = list(range(len(samples)))
        selected_idx = st.selectbox("Select Sample Index:", sample_indices)
        
        if st.button("Load & Play Dataset Sample"):
            log_to_terminal(f"Loading Dataset Sample: {target_cmd} index {selected_idx}")
            sample_data = samples[selected_idx]['audio']['array']
            selected_audio = sample_data.astype(np.float32)
            st.audio(selected_audio, sample_rate=16000)
            st.session_state['current_audio'] = selected_audio

    if st.button("🚀 Run Pipeline", use_container_width=True):
        audio_to_process = selected_audio if selected_audio is not None else st.session_state.get('current_audio')
        
        if audio_to_process is not None:
            log_to_terminal("Running Inference Pipeline...")
            with st.spinner("AI is thinking..."):
                start_time = time.time()
                result = pipeline.predict(audio_to_process)
                end_time = time.time()
                
                log_to_terminal(f"Inference complete in {end_time - start_time:.2f}s")
                log_to_terminal(f"Result: {result['prediction']} ({result['confidence']:.2%})")
                
                st.divider()
                st.subheader("Results")
                c1, c2, c3 = st.columns(3)
                c1.metric("Predicted Command", result['prediction'])
                c2.metric("Confidence", f"{result['confidence']:.2%}")
                c3.metric("Status", result['status'])
                
                st.info(f"**Transcript:** {result['transcript']}")
                st.json(result)
        else:
            st.warning("Please provide audio input first.")

with col2:
    st.subheader("RASA Assistant")
    st.image("https://rasa.com/assets/img/rasa-logo.png", width=80)
    st.write("Status: **Connected**")
    chat_history = st.container(height=300)
    
    if 'result' in locals():
        chat_history.write(f"👤 User said: {result['prediction']}")
        chat_history.write(f"🤖 RASA: Executing {result['prediction']} command...")
    else:
        chat_history.write("🤖 *Awaiting command...*")