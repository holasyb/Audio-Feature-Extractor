import os
from functools import lru_cache
from typing import Union

import librosa
import numpy as np
import torch
import torch.nn.functional as F

from .utils import exact_div

# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 80
HOP_LENGTH = 160
CHUNK_LENGTH = 30
CHUNK_LENGTH = 1
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000: number of samples in a chunk
N_FRAMES = exact_div(
    N_SAMPLES, HOP_LENGTH
)  # 3000: number of frames in a mel spectrogram input

def load_audio(file: str, sr: int = SAMPLE_RATE):
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
    audio, _ = librosa.load(file, sr=sr, dtype=np.float32)
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
    import ffmpeg # pip install ffmpeg-python
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


def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(dim=axis, index=torch.arange(length))

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array


@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int = N_MELS) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
        )
    """
    assert n_mels == 80, f"Unsupported n_mels: {n_mels}"
    with np.load(
        os.path.join(os.path.dirname(__file__), "assets", "mel_filters.npz")
    ) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor], n_mels: int = N_MELS
):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 is supported

    Returns
    -------
    torch.Tensor, shape = (80, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)

    ################### debug ###################
    mel_spec_list = []
    log_spec_list = []
    iter_num = int(len(audio) / 16000)
    for i in range(iter_num):
        audio_tmp = audio[i*16000:(i+1)*16000]
        window = torch.hann_window(N_FFT).to(audio_tmp.device)
        stft = torch.stft(audio_tmp, N_FFT, HOP_LENGTH, window=window, return_complex=True) #[201,101]

        magnitudes = stft[:, :-1].abs() ** 2    #[201,100]

        filters = mel_filters(audio_tmp.device, n_mels) # [80,201]
        mel_spec_tmp = filters @ magnitudes  # [80,201]*[201,100] --> [80,100]
        # mel_spec_list.append(mel_spec_tmp)
        log_spec = torch.clamp(mel_spec_tmp, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec_tmp = (log_spec + 4.0) / 4.0
        log_spec_list.append(log_spec_tmp)

    # mel_spec = torch.from_numpy(np.concatenate(mel_spec_list, axis=1))
    log_spec = torch.from_numpy(np.concatenate(log_spec_list, axis=1))
    ################### debug ###################


    # window = torch.hann_window(N_FFT).to(audio.device)
    # stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True) #[201,101]
    # magnitudes = stft[:, :-1].abs() ** 2
    # filters = mel_filters(audio.device, n_mels)
    # mel_spec = filters @ magnitudes  # [80,201]*[201,10] --> [80,100]
    # log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    # log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    # log_spec = (log_spec + 4.0) / 4.0

    return log_spec