import librosa
import numpy as np
import ffmpeg # pip install ffmpeg-python
import subprocess
import soundfile

def load_audio_soundfile(file: str, sr=16000):
    """
    Open an audio file and read as mono waveform, resampling as necessary
    Parameters
    ----------
    file: str
        The audio file to open
    sr: int
        The sample rate to resample the audio if necessary
    Returns
    ------- 
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    try:
        audio, _ = soundfile.read(file, dtype="float32", always_2d=True)
    except Exception as e:
        logger.error(f"load audio error, {e}")
        raise RuntimeError(f"Failed to load audio: {e}") from e
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1:
        raise ValueError(f"Audio file {file} has {audio.ndim} channels; expected mono")
    return audio

# copy from whisper
def load_audio_cmd_ffmpeg(file: str, sr=16000):
    """
    Open an audio file and read as mono waveform, resampling as necessary.
    """

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]

    try:
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except Exception as e:
        logger.error(f"load audio error, {e}")
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def load_audio_with_librosa(file_path, sr=16000):
    audio, _ = librosa.load(file_path, sr=sr, dtype=np.float32)
    return audio


def load_audio_with_ffmpeg(file: str, sr=16000):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    try:
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

if __name__ == "__main__":
    wav = "./asserts/mid.wav"
    audio_cmd_ffmpeg = load_audio_cmd_ffmpeg(wav)
    audio_librosa = load_audio_with_librosa(wav)
    audio_ffmpeg = load_audio_with_ffmpeg(wav)
    audio_soundfile = load_audio_soundfile(wav)
    print("audio_cmd_ffmpeg", audio_cmd_ffmpeg.shape, audio_cmd_ffmpeg.max(), audio_cmd_ffmpeg.min())
    print("audio_librosa", audio_librosa.shape, audio_librosa.max(), audio_librosa.min())
    print("audio_ffmpeg", audio_ffmpeg.shape, audio_ffmpeg.max(), audio_ffmpeg.min())
    print("audio_soundfile", audio_soundfile.shape, audio_soundfile.max(), audio_soundfile.min())

    error_ffmpeg = np.abs(audio_cmd_ffmpeg - audio_ffmpeg)
    error_librosa_cmd_ffmpeg = np.abs(audio_cmd_ffmpeg - audio_librosa)
    error_librosa_ffmpeg = np.abs(audio_librosa - audio_ffmpeg)
    error_soundfile = np.abs(audio_soundfile - audio_librosa)

    print("error_ffmpeg", error_ffmpeg.max(), error_ffmpeg.min())
    print("error_librosa_cmd_ffmpeg", error_librosa_cmd_ffmpeg.max(), error_librosa_cmd_ffmpeg.min())
    print("error_librosa_ffmpeg", error_librosa_ffmpeg.max(), error_librosa_ffmpeg.min())
    print("error_soundfile", error_soundfile.max(), error_soundfile.min())


