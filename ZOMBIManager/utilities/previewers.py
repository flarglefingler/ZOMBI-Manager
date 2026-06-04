from typing import List, Optional, Dict, Tuple
from io import BytesIO
import wave

def find_riff_offset(data: bytes) -> Optional[int]:
    idx = data.find(b'RIFF')
    return idx if idx != -1 else None

def extract_wav_from_son(data: bytes) -> Optional[bytes]:
    idx = find_riff_offset(data)
    return data[idx:] if idx is not None else None

def get_wav_metadata(wav_bytes: bytes) -> Optional[dict]:
    try:
        bio = BytesIO(wav_bytes)
        with wave.open(bio, "rb") as w:
            frames, rate = w.getnframes(), w.getframerate()
            channels, sampwidth = w.getnchannels(), w.getsampwidth()
            duration = frames / float(rate) if rate > 0 else 0.0
            return dict(channels=channels, sample_rate=rate,
                        sampwidth=sampwidth, frames=frames, duration=duration)
    except Exception:
        return None
