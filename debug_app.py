import streamlit as st
import os
import librosa
import numpy as np
import time
import torch
from datasets import load_from_disk
import requests
import json

from backend.pipeline import SpeakPipeline

# ---------------- ENV ----------------
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

st.set_page_config(page_title="Speak Prototype", layout="wide")

# ---------------- OLLAMA CONFIG ----------------
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "deepseek-coder:latest" 

SYSTEM_PROMPT = """
You are a Robot Command Assistant. 
Take the user's input and map it to exactly one of these commands: [FORWARD, BACKWARD, LEFT, RIGHT, STOP, GREETING].
Respond ONLY with the command name. No punctuation.
"""

# ---------------- LOG ----------------
def log(msg):
    print(f"[SYSTEM LOG] {time.strftime('%H:%M:%S')} - {msg}", flush=True)

# ---------------- OLLAMA CALL (Replaces Rasa) ----------------
def call_ollama(message):
    if not message or len(message.strip()) == 0:
        return "UNKNOWN"
    try:
        payload = {
            "model": MODEL_NAME,
            "prompt": f"{SYSTEM_PROMPT}\nUser input: {message}\nCommand:",
            "stream": False
        }
        # Reduced timeout to 10s for better UI feel
        response = requests.post(OLLAMA_URL, json=payload, timeout=10)
        result = response.json()
        return result.get("response", "UNKNOWN").strip().upper()
    except Exception as e:
        return f"OFFLINE (Check Ollama App)"

# ---------------- PARSER ----------------
def parse_result(result):
    if not isinstance(result, dict):
        return "N/A", "0%", "unknown", ""

    prediction = result.get("command") or result.get("prediction") or "N/A"
    confidence = result.get("confidence", result.get("score", 0))
    status = result.get("status", "success")
    transcript = result.get("transcript", "")

    try:
        confidence_display = f"{float(confidence):.2%}"
    except:
        confidence_display = str(confidence)

    return prediction, confidence_display, status, transcript

# ---------------- PIPELINE ----------------
def load_pipeline():
    log("Loading pipeline...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = SpeakPipeline(
        device=device,
        whisper_model_name="openai/whisper-medium",
        lora_path="models/checkpoint-1488",
        mapper_model_path="models/artifacts/mapper_model.joblib",
        label_encoder_path="models/artifacts/label_encoder.joblib"
    )
    log("Pipeline loaded")
    return pipe

# ---------------- DATASET ----------------
def load_dataset():
    log("Loading dataset...")
    path = "dataset/easycall_mvp_dataset"
    ds = load_from_disk(path)
    if "text" not in ds.column_names:
        raise ValueError("Dataset missing 'text' column")
    commands = list(set(ds["text"]))
    log(f"Dataset loaded with {len(commands)} commands")
    return ds, commands

# ---------------- STATE ----------------
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- INIT ----------------
if st.button("🔄 Initialize System"):
    try:
        # Check if Ollama is running first
        requests.get("http://localhost:11434/", timeout=2)
        
        with st.spinner("Loading models into GPU..."):
            st.session_state.pipeline = load_pipeline()
            dataset, commands = load_dataset()
            st.session_state.dataset = dataset
            st.session_state.commands = commands
        st.success("System initialized! Ollama & Whisper ready.")
    except requests.exceptions.ConnectionError:
        st.error("OLLAMA ERROR: Please start the Ollama application from your taskbar.")
    except Exception as e:
        st.error(f"INIT ERROR: {e}")

# ---------------- UI ----------------
st.title("🗣️ Speak Prototype")

col1, col2 = st.columns([2, 1])

# ================= LEFT PANEL (Audio) =================
with col1:
    st.subheader("Audio Input")
    input_mode = st.radio("Choose Input:", ["Upload Audio", "Dataset"], horizontal=True)
    selected_audio = None

    if input_mode == "Upload Audio":
        uploaded_file = st.file_uploader("Upload .wav", type=["wav"])
        if uploaded_file:
            audio_bytes, _ = librosa.load(uploaded_file, sr=16000)
            selected_audio = audio_bytes
            st.audio(uploaded_file)
    else:
        if st.session_state.get("dataset") is None:
            st.warning("Initialize system first")
        else:
            target_cmd = st.selectbox("Command", st.session_state.commands)
            indices = [i for i, x in enumerate(st.session_state.dataset) if x["text"] == target_cmd]
            selected_idx = st.selectbox("Sample Index", indices)

            if st.button("Load Sample"):
                sample = st.session_state.dataset[selected_idx]["audio"]["array"]
                selected_audio = np.array(sample, dtype=np.float32)
                st.audio(selected_audio, sample_rate=16000)
                st.session_state["current_audio"] = selected_audio

    if st.button("🚀 Run Pipeline", use_container_width=True):
        audio = selected_audio if selected_audio is not None else st.session_state.get("current_audio")
        if audio is None or st.session_state.pipeline is None:
            st.warning("Initialize system and provide audio")
        else:
            with st.spinner("Running inference..."):
                result = st.session_state.pipeline.predict(audio)
                prediction, conf, status, transcript = parse_result(result)
                
                # Ollama Intelligent Mapping
                ollama_intent = call_ollama(transcript)
                
                # Format Response: "Yes boss, [My Model Prediction] is recognized. Initiating [Ollama Command]."
                final_response = f"Yes boss, {prediction} is recognized. Initiating {ollama_intent}."
                
                st.session_state['assistant_reply'] = final_response
                st.session_state['last_result'] = transcript
                
                # Sync to chat
                st.session_state.chat_history.append({"role": "user", "content": f"[Audio] {transcript}"})
                st.session_state.chat_history.append({"role": "assistant", "content": final_response})

                st.subheader("Results")
                c1, c2, c3 = st.columns(3)
                c1.metric("Model Prediction", prediction)
                c2.metric("Confidence", conf)
                c3.metric("Status", status)
                st.info(f"Transcript: {transcript}")


# ================= RIGHT PANEL (Assistant & Chat) =================
with col2:
    st.subheader("🤖 Assistant Status")
    
    if 'assistant_reply' in st.session_state:
        st.info(f"**Last Action:** {st.session_state['last_result']}")
        st.success(f"**Status:** {st.session_state['assistant_reply']}")
    else:
        st.write("Awaiting input...")

    st.divider()
    st.subheader("💬 Chat")

    if user_text := st.chat_input("Type your command..."):
        st.session_state.chat_history.append({"role": "user", "content": user_text})
        
        with st.spinner("Ollama thinking..."):
            ollama_intent = call_ollama(user_text)
            # For text input, we use the input as the "recognized" part
            final_reply = f"Yes boss, {user_text} is recognized. Initiating {ollama_intent}."
            st.session_state.chat_history.append({"role": "assistant", "content": final_reply})

    for msg in reversed(st.session_state.chat_history):
        if msg["role"] == "user":
            st.write(f"👤 **You:** {msg['content']}")
        else:
            st.info(f"🤖 **Assistant:** {msg['content']}")