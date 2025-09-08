from pydub import AudioSegment
import os

def convert_mp3_to_wav(mp3_path):
    if not os.path.exists(mp3_path):
        raise FileNotFoundError(f"MP3 file not found at: {mp3_path}")
    
    wav_path = os.path.splitext(mp3_path)[0] + '.wav'
    try:
        audio = AudioSegment.from_mp3(mp3_path)
        audio.export(wav_path, format="wav")
        print(f"Converted {mp3_path} to {wav_path}")
        return wav_path
    except Exception as e:
        print(f"Error converting file: {e}")
        raise