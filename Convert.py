import os
from pydub import AudioSegment
import subprocess
import ffmpeg
from pathlib import Path

def check_ffmpeg():
    """Check if ffmpeg is installed and accessible."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: ffmpeg is not installed or not found in PATH. Install ffmpeg.")
        return False

def verify_audio_stream(audio_path):
    """Verify if the audio file has an audio stream."""
    try:
        probe = ffmpeg.probe(audio_path)
        has_audio = any(stream['codec_type'] == 'audio' for stream in probe['streams'])
        if not has_audio:
            print(f"Warning: {audio_path} has no audio stream.")
        return has_audio
    except ffmpeg.Error as e:
        print(f"Error probing {audio_path}: {e.stderr.decode()}")
        return False

def convert_audio_to_wav():
    """
    Convert MP4 or M4A audio file(s) to WAV format using an interactive input path.
    Output directory is hardcoded to 'data/newwav' and sample rate to 44100 Hz.
    """
    output_dir = "data\DAY 4"
    sample_rate = 44100
    
    if not check_ffmpeg():
        raise RuntimeError("ffmpeg is required for audio conversion.")
    
    
    input_path = input("Enter the path to the MP4/M4A audio file or directory containing MP4/M4A audio files: ").strip()
    
    
    if not os.path.exists(input_path):
        raise ValueError(f"Input path {input_path} does not exist.")
    
   
    os.makedirs(output_dir, exist_ok=True)
    
    
    converted_files = []
    if os.path.isfile(input_path):
        if input_path.lower().endswith(('.mp4', '.m4a')):
            wav_path = process_single_file(input_path, output_dir, sample_rate)
            if wav_path:
                converted_files.append(wav_path)
        else:
            print(f"Error: {input_path} is not an MP4 or M4A file")
    elif os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            for filename in files:
                if filename.lower().endswith(('.mp4', '.m4a')):
                    audio_path = os.path.join(root, filename)
                    relative_path = os.path.relpath(root, input_path)
                    output_subdir = os.path.join(output_dir, relative_path)
                    os.makedirs(output_subdir, exist_ok=True)
                    wav_path = process_single_file(audio_path, output_subdir, sample_rate)
                    if wav_path:
                        converted_files.append(wav_path)
    else:
        print(f"Error: {input_path} is not a valid file or directory")
    
    print(f"Converted {len(converted_files)} files")
    return converted_files

def process_single_file(audio_path, output_dir, sample_rate):
    """
    Convert a single MP4 or M4A file to WAV format.
    
    Args:
        audio_path (str): Path to input MP4 or M4A file
        output_dir (str): Directory to save WAV file
        sample_rate (int): Target sample rate for WAV file
    """
    try:
        if not verify_audio_stream(audio_path):
            return None
        
        #Determine input format based on file extension
        file_ext = os.path.splitext(audio_path)[1].lower()
        input_format = "mp4" if file_ext == ".mp4" else "m4a"
        
       
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        wav_path = os.path.join(output_dir, f"{base_name}.wav")
        
        
        print(f"Converting {audio_path} to {wav_path}...")
        audio = AudioSegment.from_file(audio_path, format=input_format)
        audio = audio.set_frame_rate(sample_rate)
        audio.export(wav_path, format="wav")
        
        print(f"Successfully converted {audio_path} to {wav_path}")
        return wav_path
        
    except Exception as e:
        print(f"Error converting {audio_path}: {str(e)}")
        return None

if __name__ == "__main__":
    convert_audio_to_wav()