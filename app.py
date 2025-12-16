import streamlit as st
import cv2
from face_emotion import detect_emotion
from text_sentiment import analyze_text
from voice_stress import analyze_voice
from federated import federated_training

st.set_page_config(layout="wide")
st.title("Privacy-Preserving Multimodal AI System")
st.subheader("Student Name: Sanjana")

col1, col2, col3 = st.columns(3)

# ---------------- FACE MODULE ----------------
with col1:
    st.header("Facial Emotion Detection")
    run = st.checkbox("Start Webcam")
    frame_window = st.image([])

    cam = cv2.VideoCapture(0)
    emotion = "Neutral"

    if run:
        ret, frame = cam.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        emotion, confidence = detect_emotion(frame)
        frame_window.image(frame)
        st.success(f"Emotion: {emotion} ({confidence:.2f}%)")

# ---------------- TEXT & VOICE ----------------
with col2:
    st.header("Text & Voice Analysis")

    text = st.text_input("Enter text")
    sentiment, polarity = analyze_text(text)
    st.info(f"Text Sentiment: {sentiment}")

    audio = st.file_uploader("Upload Voice (.wav)", type=["wav"])
    stress = "Low"

    if audio:
        with open("sample.wav", "wb") as f:
            f.write(audio.read())

        stress, score = analyze_voice("sample.wav")
        st.warning(f"Voice Stress: {stress}")

    if emotion == "happy" and stress == "Low":
        risk = "Low Risk"
    elif stress == "High":
        risk = "High Risk"
    else:
        risk = "Medium Risk"

    st.success(f"Final Risk Level: {risk}")

# ---------------- FEDERATED ----------------
with col3:
    st.header("Federated Learning")
    logs = federated_training()
    for log in logs:
        st.write(log)
