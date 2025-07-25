import streamlit as st
from drowsiness_detector import run_drowsiness_detection

st.set_page_config(page_title="No Sleep Alarm", layout="centered")
st.title("🛑 No Sleep Alarm - Drowsiness Detection")

st.markdown("""
This app uses your **webcam**, **deep learning**, and **MediaPipe** to detect drowsiness in real-time.  
If your eyes remain closed for several seconds, an alarm will sound.
""")

if st.button("Start Drowsiness Detection"):
    st.warning("Press 'Q' in the webcam window to stop the detection.")
    run_drowsiness_detection()