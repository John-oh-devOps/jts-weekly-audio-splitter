import os
import re
import zipfile
import tempfile
import whisper
import streamlit as st
from pydub import AudioSegment
from pydub.silence import split_on_silence

# Configuration
MIN_SILENCE_LEN = 2000  # 2 seconds
SILENCE_THRESH = -50    # Silence threshold in dBFS
KEEP_SILENCE = 300      # Keep 300ms buffer at start/end of cuts

def sanitize_filename(text: str, max_words: int = 5) -> str:
    """Clean transcribed text to create a safe, filesystem-friendly filename."""
    cleaned = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE).strip().lower()
    if not cleaned:
        return ""
    words = cleaned.split()[:max_words]
    return "_".join(words)

# Cache the model so it doesn't reload every time you click a button
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("tiny")

def process_audio(audio_file_path, output_dir):
    model = load_whisper_model()
    sound = AudioSegment.from_file(audio_file_path)
    
    chunks = split_on_silence(
        sound,
        min_silence_len=MIN_SILENCE_LEN,
        silence_thresh=SILENCE_THRESH,
        keep_silence=KEEP_SILENCE
    )

    if not chunks:
        return []

    os.makedirs(output_dir, exist_ok=True)
    used_filenames = set()
    generated_files = []

    # Streamlit progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, chunk in enumerate(chunks, start=1):
        chunk_for_whisper = chunk.set_frame_rate(16000).set_channels(1)
        temp_filename = os.path.join(output_dir, f"temp_{i}.wav")
        
        try:
            chunk_for_whisper.export(temp_filename, format="wav")
            result = model.transcribe(temp_filename, fp16=False)
            transcript = result.get("text", "").strip()

            clean_snippet = sanitize_filename(transcript, max_words=5)

            if not clean_snippet:
                clean_name = f"take_{i:03d}"
            else:
                clean_name = f"{i:03d}_{clean_snippet}"

            final_name = clean_name
            counter = 1
            while final_name in used_filenames:
                final_name = f"{clean_name}_v{counter}"
                counter += 1
            used_filenames.add(final_name)

            final_path = os.path.join(output_dir, f"{final_name}.wav")
            chunk.export(final_path, format="wav")
            
            # Store info for the UI
            generated_files.append({
                "filename": f"{final_name}.wav",
                "path": final_path,
                "transcript": transcript
            })

        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
        
        # Update progress bar
        progress_bar.progress(i / len(chunks))
        status_text.text(f"Processed {i}/{len(chunks)} segments...")

    return generated_files

def main():
    # Set page layout to have a distinct sidebar on the left
    st.set_page_config(page_title="Audio Splitter", layout="wide")
    
    st.title("✂️ Auto Audio Splitter & Namer")
    st.markdown("Upload an audio file. The app will split it by silence, transcribe it using Whisper, and let you download the named segments.")
    
    # Left Sidebar for results
    st.sidebar.header("📁 Generated Files")
    
    # Main area for upload
    uploaded_file = st.file_uploader("Choose an audio file", type=["m4a", "mp3", "wav", "ogg"])
    
    if uploaded_file is not None:
        if st.button("Process Audio", type="primary"):
            # Use a temporary directory so files are cleaned up automatically from your system
            with tempfile.TemporaryDirectory() as temp_dir:
                input_path = os.path.join(temp_dir, uploaded_file.name)
                
                # Save the uploaded file to disk temporarily
                with open(input_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                output_dir = os.path.join(temp_dir, "output")
                
                with st.spinner("Processing audio... (This might take a minute)"):
                    generated_files = process_audio(input_path, output_dir)
                
                if generated_files:
                    st.success(f"Successfully created {len(generated_files)} audio files!")
                    
                    # Package all files into a ZIP
                    zip_path = os.path.join(temp_dir, "split_takes.zip")
                    with zipfile.ZipFile(zip_path, 'w') as zipf:
                        for file_info in generated_files:
                            zipf.write(file_info['path'], arcname=file_info['filename'])
                    
                    # Read the ZIP into memory for the download button
                    with open(zip_path, "rb") as f:
                        zip_data = f.read()
                    
                    # Add Download Button to Sidebar
                    st.sidebar.download_button(
                        label="⬇️ Download All (ZIP)",
                        data=zip_data,
                        file_name="split_takes.zip",
                        mime="application/zip",
                        type="primary"
                    )
                    
                    st.sidebar.markdown("---")
                    
                    # List all files on the left sidebar
                    for item in generated_files:
                        st.sidebar.markdown(f"**{item['filename']}**")
                        st.sidebar.caption(f"_{item['transcript']}_")
                else:
                    st.error("No speech segments detected. The audio might be too quiet or lack pauses.")

if __name__ == "__main__":
    main()