import streamlit as st
from openai import OpenAI
import numpy as np
import tempfile
import os
from pydub import AudioSegment
import wave
from audio_recorder_streamlit import audio_recorder

# Set page config
st.set_page_config(page_title="Speech to Text Converter", page_icon="🎤")

# Initialize session state
if 'api_key_configured' not in st.session_state:
    st.session_state.api_key_configured = False
if 'client' not in st.session_state:
    st.session_state.client = None

def configure_api_key():
    api_key = st.sidebar.text_input("Enter OpenAI API Key:", type="password")
    if api_key:
        st.session_state.client = OpenAI(api_key=api_key)
        st.session_state.api_key_configured = True
        return True
    return False

def convert_audio_to_wav(input_file, output_file):
    """Convert various audio formats to WAV"""
    audio = AudioSegment.from_file(input_file)
    audio.export(output_file, format="wav")

def transcribe_audio(audio_file):
    """Transcribe audio using OpenAI Whisper API"""
    try:
        with open(audio_file, "rb") as file:
            transcript = st.session_state.client.audio.transcriptions.create(
                model="whisper-1",
                file=file
            )
        return transcript.text
    except Exception as e:
        st.error(f"Error during transcription: {str(e)}")
        return None

def main():
    st.title("🎤 Speech to Text Converter")
    st.write("Convert speech to text using OpenAI's Whisper API")

    # Configure API key in sidebar
    if not configure_api_key():
        st.warning("Please enter your OpenAI API key in the sidebar to continue.")
        return

    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["Upload Audio", "Record Microphone"])

    with tab1:
        st.header("Upload Audio File")
        uploaded_file = st.file_uploader("Choose an audio file", 
                                       type=['wav', 'mp3', 'wma'])
        
        if uploaded_file:
            with st.spinner("Processing audio file..."):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    input_path = tmp_file.name

                # Convert to WAV if needed
                if not uploaded_file.name.endswith('.wav'):
                    wav_path = input_path + '.wav'
                    convert_audio_to_wav(input_path, wav_path)
                else:
                    wav_path = input_path

                # Transcribe
                transcript = transcribe_audio(wav_path)
                
                # Clean up temporary files
                os.unlink(input_path)
                if input_path != wav_path:
                    os.unlink(wav_path)

                if transcript:
                    st.success("Transcription Complete!")
                    st.write(transcript)

    with tab2:
        st.header("Record from Microphone")
        
        # Audio recorder
        audio_bytes = audio_recorder()
        
        if audio_bytes:
            # Save the recorded audio to a temporary WAV file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_bytes)
                wav_path = tmp_file.name

            with st.spinner("Processing recording..."):
                # Transcribe
                transcript = transcribe_audio(wav_path)
                
                # Clean up
                os.unlink(wav_path)

                if transcript:
                    st.success("Transcription Complete!")
                    st.write(transcript)

if __name__ == "__main__":
    main()
