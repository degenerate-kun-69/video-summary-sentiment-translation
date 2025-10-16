"""
speech to text app
"""

import os
import tempfile
import traceback
from typing import Optional, Dict, Any

import numpy as np
import soundfile as sf

# Safe imports
try:
    import torch
except ImportError:
    torch = None


try:
    import whisper
except ImportError:
    whisper = None


def detect_environment() -> Dict[str, Any]:
    """
    check for any available gpu accel fw
    """
    cuda = False
    hip = False
    torch_available = False

    if torch is not None:
        torch_available = True
        try:
            cuda = torch.cuda.is_available()
        except Exception:
            cuda = False
        hip = getattr(torch.version, "hip", None) is not None

    return {
        "torch_available": torch_available,
        "cuda_available": cuda,
        "hip_available": hip,
        "whisper_available": whisper is not None,
        "device_type": "cuda" if cuda else "rocm" if hip else "cpu"
    }


class HindiTranscriber:
    """
    Main transcriber class — auto-selects backend for Whisper models.

    Example:
    ---------
    >>> from transcribe import HindiTranscriber
    >>> transcriber = HindiTranscriber(model_name="medium")
    >>> text = transcriber.transcribe("speech-file.wav")
    >>> print(text)
    """

    def __init__(self, model_name: str = "small", language: str = "hi"):
        self.model_name = model_name
        self.language = language
        self.env = detect_environment()
        self.device = self.env["device_type"]
        self.model = None
        self.backend = None
        self._load_model()

    def _load_model(self):
        """
        Loads the appropriate whisper model based on available acceleration.
        """

        # 1️⃣ Try faster-whisper (CUDA only)
        if self.env["cuda_available"] and self.env["faster_whisper_available"]:
            try:
                print("[WhisperBackend] Using faster-whisper on CUDA")
                self.model = FasterWhisperModel(self.model_name, device="cuda", compute_type="float16")
                self.backend = "faster-whisper"
                return
            except Exception as e:
                print("[WhisperBackend] Faster-whisper failed:", e)
                traceback.print_exc()

        # 2️⃣ Try OpenAI whisper with PyTorch (ROCm/CUDA/CPU)
        if self.env["torch_available"] and self.env["whisper_available"]:
            try:
                device = "cuda" if self.env["cuda_available"] or self.env["hip_available"] else "cpu"
                print(f"[WhisperBackend] Using OpenAI whisper on {device}")
                self.model = whisper.load_model(self.model_name, device=device)
                self.backend = "openai-whisper"
                return
            except Exception as e:
                print("[WhisperBackend] OpenAI whisper load failed:", e)
                traceback.print_exc()

        # 3️⃣ CPU fallback
        if self.env["whisper_available"]:
            print("[WhisperBackend] Falling back to CPU whisper")
            self.model = whisper.load_model(self.model_name, device="cpu")
            self.backend = "openai-whisper"
            return

        raise RuntimeError("No valid whisper backend found. Install whisper or faster-whisper with torch.")

    # -----------------------------------------------------------------------

    def transcribe(self, audio_path: str, task: str = "transcribe") -> str:
        """
        Transcribes a given audio file path and returns the text output.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if self.backend == "faster-whisper":
            segments, _ = self.model.transcribe(audio_path, language=self.language, task=task)
            return " ".join([s.text.strip() for s in segments])

        elif self.backend == "openai-whisper":
            result = self.model.transcribe(audio_path, language=self.language, task=task)
            return result.get("text", "").strip()

        else:
            raise RuntimeError("Transcriber backend not initialized properly.")

    # -----------------------------------------------------------------------

    def transcribe_numpy(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Accepts a numpy float32 array (mono) and performs transcription.
        """
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, audio_data, sample_rate)
        tmp.close()

        try:
            text = self.transcribe(tmp.name)
            return text
        finally:
            os.remove(tmp.name)

    # -----------------------------------------------------------------------

    def info(self) -> Dict[str, Any]:
        """Returns model environment details."""
        return {
            "backend": self.backend,
            "device": self.device,
            "env": self.env,
            "model_name": self.model_name,
        }

