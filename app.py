import streamlit as st
from colorthief import ColorThief
import cv2
import os
import numpy as np
import colorsys

# import gradio as gr

# --- Simple Theme Styling ---
st.markdown(
    """
    <style>
    .stApp {
        background-color: #83EEFF;
        font-family: 'Segoe UI', sans-serif;
    }
    h1 {
        color: #003344;
        text-align: center;
    }
    .uploadedFile {
        border-radius: 10px;
        padding: 10px;
    }
    .mood-box {
        background-color: white;
        padding: 12px;
        border-radius: 8px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        color: #003344;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
        margin-top: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Mood detection from color palette ---
def detect_mood_from_palette(palette):
    hsv_palette = [colorsys.rgb_to_hsv(r / 255, g / 255, b / 255) for r, g, b in palette]
    hues = [h for h, s, v in hsv_palette]
    sats = [s for h, s, v in hsv_palette]
    vals = [v for h, s, v in hsv_palette]

    if all(v < 0.3 for v in vals):
        return "angry"
    if all(v > 0.8 for v in vals) and all(s < 0.3 for s in sats):
        return "dreamy"
    if any(0.0 < h < 0.15 for h in hues):
        return "angry"
    if any(0.15 < h < 0.25 for h in hues):
        return "happy"
    if any(0.25 < h < 0.45 for h in hues):
        return "calm"
    if any(0.45 < h < 0.7 for h in hues):
        return "sad"
    if any(0.7 < h < 0.9 for h in hues):
        return "dreamy"
    return "happy"

# --- Basic facial emotion detection using OpenCV ---
def detect_facial_mood(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
    eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None  # No face detected

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]

        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        if len(smiles) > 0:
            return "happy"

        eyes = eyes_cascade.detectMultiScale(roi_gray)
        if len(eyes) == 0:
            return "calm"

    return "neutral"

# --- Analyze image ---
def analyze_image(image_path):
    face_mood = detect_facial_mood(image_path)
    if face_mood and face_mood != "neutral":
        return face_mood
    else:
        palette = ColorThief(image_path).get_palette(color_count=5)
        return detect_mood_from_palette(palette)

# --- Analyze video ---
def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < 5:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    moods = []
    for frame in frames:
        temp_frame_path = "temp_frame.jpg"
        cv2.imwrite(temp_frame_path, frame)
        moods.append(analyze_image(temp_frame_path))
        os.remove(temp_frame_path)

    return max(set(moods), key=moods.count) if moods else "happy"

# --- Streamlit UI ---
st.title("PicTune - AI Mood Detector ")
uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file is not None:
    temp_file = "temp_upload"
    with open(temp_file, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if uploaded_file.type.startswith("image/"):
        mood = analyze_image(temp_file)
        st.image(temp_file, caption="Uploaded Image")
    elif uploaded_file.type.startswith("video/"):
        mood = analyze_video(temp_file)
        st.video(temp_file)

    st.markdown(f"<div class='mood-box'>🎯 Detected Mood: {mood.capitalize()}</div>", unsafe_allow_html=True)

    audio_files = {
        "happy": "pictune-assets/music/happy.mp3",
        "calm": "pictune-assets/music/calm.mp3",
        "sad": "pictune-assets/music/sad.mp3",
        "angry": "pictune-assets/music/angry.mp3",
        "dreamy": "pictune-assets/music/dreamy.mp3"
    }

    if mood in audio_files:
        st.audio(audio_files[mood])

    os.remove(temp_file)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #83EEFF;
        font-family: 'Segoe UI', sans-serif;
    }
    h1 {
        color: black !important;
        text-align: center;
        font-size: 3rem !important; /* Bigger title */
        font-weight: 800 !important;
    }
    /* Subtitle: file uploader label */
    .stFileUploader label {
        color: black !important;
        font-size: 1.3rem !important;
        font-weight: 600 !important;
    }
    /* Upload box styling */
    .stFileUploader {
        transform: scale(1.1); /* Enlarge box */
    }
    .uploadedFile {
        border-radius: 10px;
        padding: 10px;
    }
    .mood-box {
        background-color: white;
        padding: 12px;
        border-radius: 8px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        color: #003344;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
        margin-top: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 5000)))
