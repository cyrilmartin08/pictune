import streamlit as st
from colorthief import ColorThief
import cv2
import os
import numpy as np
import colorsys

# --- Simple Theme Styling for better mobile view ---
st.markdown(
    """
    <style>
    /* Full page styling */
    .stApp {
        background-color: #83EEFF;
        font-family: 'Segoe UI', sans-serif;
        display: flex;
        flex-direction: column;
        min-height: 100vh; /* Make sure the app takes at least the full viewport height */
        padding: 1rem; /* A little extra padding for the whole app */
    }
    
    /* Centralize the block container to handle spacing */
    .block-container {
        padding-top: 1rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        flex-grow: 1; /* Allow content to grow and push the footer down */
    }

    /* Main title styling */
    h1 {
        color: black !important;
        text-align: center;
        font-size: 3rem !important;
        font-weight: 800 !important;
        margin-top: 0 !important; /* Remove default top margin */
        padding-top: 20px; /* Add padding to the top of the title */
        margin-bottom: 20px;
    }
    /* Subtitle: file uploader label */
    .stFileUploader label {
        color: black !important;
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        text-align: center; /* Centralize the label text */
        width: 100%;
        display: block;
    }
    /* Upload box styling */
    .stFileUploader > div > div {
        display: flex;
        justify-content: center; /* Centralize the drag-and-drop box */
    }
    .stFileUploader {
        transform: scale(1.1);
        margin: 20px 0;
        padding: 0 10px; /* Add some space on the sides for a better look */
    }
    /* File uploader button style */
    .stFileUploader > div > div > div > button {
        background-color: white !important;
        color: #003344 !important;
        border: 2px solid #003344 !important;
    }
    /* Uploaded file details */
    .uploadedFile {
        border-radius: 10px;
        padding: 10px;
    }
    /* Mood detection box style */
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
    /* General responsive images/videos */
    .stImage > img, .stVideo > video {
        max-width: 100%;
        height: auto;
        border-radius: 10px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    }
    /* Footer styling */
    .footer {
        text-align: center;
        font-size: 0.9rem;
        color: #003344;
        margin-top: auto; /* Push the footer to the bottom */
        padding-top: 20px;
        padding-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Mood detection from color palette ---
def detect_mood_from_palette(palette):
    """
    Analyzes a color palette to determine a mood based on simple heuristics.
    This function has been expanded to include more moods.
    """
    hsv_palette = [colorsys.rgb_to_hsv(r / 255, g / 255, b / 255) for r, g, b in palette]
    hues = [h for h, s, v in hsv_palette]
    sats = [s for h, s, v in hsv_palette]
    vals = [v for h, s, v in hsv_palette]

    avg_sat = np.mean(sats)
    avg_val = np.mean(vals)

    # Heuristics for the new moods
    if avg_sat < 0.2 and avg_val > 0.5:
        return "neutral"
    if avg_sat > 0.7 and avg_val > 0.7:
        return "surprised"
    if avg_val < 0.3:
        return "fearful"  # Dark, low-value colors
    if any(0.2 < h < 0.35 for h in hues) and avg_sat > 0.4:
        return "disgusted" # Greenish hues

    # Original moods logic
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

    return "neutral" # A better fallback than "happy"

# --- Basic facial emotion detection using OpenCV ---
def detect_facial_mood(image_path):
    """
    Detects basic facial emotions (happy, calm, neutral) using Haar cascades.
    Note: More complex emotions like surprise or fear require a more advanced
    model than simple cascades.
    """
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
    """
    Analyzes a single image for mood, prioritizing facial detection.
    """
    face_mood = detect_facial_mood(image_path)
    if face_mood and face_mood != "neutral":
        return face_mood
    else:
        palette = ColorThief(image_path).get_palette(color_count=5)
        return detect_mood_from_palette(palette)

# --- Analyze video ---
def analyze_video(video_path):
    """
    Analyzes a video by sampling a few frames and determining the most common mood.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    # Analyze a few frames to save processing time
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

    # Return the most frequently detected mood
    return max(set(moods), key=moods.count) if moods else "neutral"

# --- Streamlit UI ---
st.title("PicTune - AI Mood & Music Matcher 🎵")
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

    # Dictionary of moods and corresponding placeholder audio files.
    # Replace these with your own mp3 files in the 'pictune-assets/music' directory.
    audio_files = {
        "happy": "pictune-assets/music/happy.mp3",
        "calm": "pictune-assets/music/calm.mp3",
        "sad": "pictune-assets/music/sad.mp3",
        "angry": "pictune-assets/music/angry.mp3",
        "dreamy": "pictune-assets/music/dreamy.mp3",
        "surprised": "pictune-assets/music/surprised.mp3",
        "fearful": "pictune-assets/music/fearful.mp3",
        "neutral": "pictune-assets/music/neutral.mp3",
        "disgusted": "pictune-assets/music/disgusted.mp3"
    }

    if mood in audio_files and os.path.exists(audio_files[mood]):
        st.audio(audio_files[mood])
    elif mood in audio_files:
        st.warning(f"Audio file for '{mood}' not found. Please add '{audio_files[mood]}' to your project.")

    os.remove(temp_file)

# --- Footer ---
st.markdown(
    """
    <div class="footer">
        Made with ❤️ by Cyril Martin
    </div>
    """,
    unsafe_allow_html=True
)
