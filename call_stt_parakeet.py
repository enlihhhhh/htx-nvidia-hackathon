import subprocess
import tempfile
import os
from dotenv import load_dotenv

def riva_offline_transcribe(audio_bytes: str, api_key: str, riva_client_path: str) -> str:
    """
    Calls the NVIDIA Riva offline ASR client via subprocess.

    :param audio_bytes: Raw WAV audio bytes.
    :param api_key: API key for Riva NGC access.
    :param riva_client_path: Path to transcribe_file_offline.py
    :return: Transcription text (stdout from Riva client)
    """
    # TODO <Ros> : Update function to take in wav file/audio sample somehow.
    # Write audio to a temporary WAV file
    # with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
    #    tmp_file.write(audio_bytes)
    #    tmp_file_path = tmp_file.name

    # Build command
    tmp_file_path = audio_bytes

    # Using Try API
    cmd = [
        "python", riva_client_path,
        "--server", "grpc.nvcf.nvidia.com:443",
        "--use-ssl",
        "--metadata", "function-id", "d3fe9151-442b-4204-a70d-5fcc597fd610",
        "--metadata", "authorization", f"Bearer {api_key}",
        "--language-code", "en-US",
        "--word-time-offsets",
        "--automatic-punctuation",
        "--input-file", tmp_file_path
    ]

    try:
        # Execute subprocess
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        transcription = result.stdout
    except subprocess.CalledProcessError as e:
        transcription = e.stderr
    # TODO <Ros> : reinstate del of tmpfile
    #finally:
        # Clean up temporary audio file
        #os.remove(tmp_file_path)

    return transcription


# Test usage
if __name__ == "__main__":
    load_dotenv()
    riva_client_path = "stt_client.py"
    api_key = os.environ.get("RIVA_API_KEY")  

    print(api_key)
    # Load example WAV file
    # with open("example.wav", "rb") as f:
    #    audio_bytes = f.read()
    
    audio_bytes = "sample.wav"

    transcript = riva_offline_transcribe(audio_bytes, api_key, riva_client_path)
    print(transcript)
