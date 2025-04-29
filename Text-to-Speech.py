import os
import argparse
import platform
import subprocess
import pyttsx3
from tempfile import NamedTemporaryFile
import wave
import contextlib
import shutil
from get_caption import generate_caption_for_image

# For MP3 conversion
try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None  # pydub requires ffmpeg installed on the system

def speak_text_offline(text: str, lang: str = 'en', save_path: str = None) -> None:
    """
    Convert text to speech using pyttsx3 (offline), play it immediately,
    and optionally save to a WAV or MP3 audio file.
    """
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)

    # Generate speech to a temporary WAV file
    with NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        tmp_wav = tmp.name
    engine.save_to_file(text, tmp_wav)
    engine.runAndWait()

    # Verify the generated WAV is valid
    try:
        with contextlib.closing(wave.open(tmp_wav, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            if duration < 0.1:
                print("Error: Generated audio too short or empty.")
                os.remove(tmp_wav)
                return
    except Exception as e:
        print(f"Error: Failed to generate valid audio WAV: {e}")
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)
        return

    # Save or play
    if save_path:
        base, ext = os.path.splitext(save_path)
        ext = ext.lower()
        if ext == '.mp3':
            if AudioSegment is None:
                print("Error: pydub not installed; cannot convert to MP3. Install pydub and ffmpeg.")
                final_path = base + '.wav'
                shutil.move(tmp_wav, final_path)
                print(f"Audio saved to: {final_path}")
                return
            # Convert WAV to MP3
            audio = AudioSegment.from_wav(tmp_wav)
            final_path = base + '.mp3'
            audio.export(final_path, format='mp3')
            os.remove(tmp_wav)
            print(f"Audio saved to: {final_path}")
        elif ext == '.wav':
            final_path = base + '.wav'
            shutil.move(tmp_wav, final_path)
            print(f"Audio saved to: {final_path}")
        else:
            final_path = base + '.wav'
            shutil.move(tmp_wav, final_path)
            print(f"Unsupported extension; audio saved as WAV: {final_path}")
    else:
        system = platform.system()
        try:
            if system == "Windows":
                os.startfile(tmp_wav)
            elif system == "Darwin":  # macOS
                subprocess.run(["afplay", tmp_wav], check=True)
            else:  # Linux/Unix
                subprocess.run(["aplay", tmp_wav], check=True)
        except Exception as e:
            print(f"Could not play audio: {e}")
            print(f"Audio file is at: {tmp_wav}")
        finally:
            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)


def main():
    parser = argparse.ArgumentParser(description="Generate image caption and speak it aloud (offline)")
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("--lang", default="en", help="Language for speech (ignored by pyttsx3)")
    parser.add_argument("--save", help="Optional path to save the spoken audio (e.g. output.mp3 or output.wav)")
    args = parser.parse_args()

    # Generate caption
    caption = generate_caption_for_image(args.image)
    print("Caption:", caption)

    # Speak offline and optionally save
    speak_text_offline(caption, lang=args.lang, save_path=args.save)


if __name__ == "__main__":
    main()