"""Speech-to-text (STT) provider abstraction for Saido Agent voice pipeline.

Provides a provider-based STT architecture with lazy model loading:

  - ``FasterWhisperSTT`` -- local, offline, fastest (pip install faster-whisper)
  - ``DeepgramSTT``      -- cloud, real-time (pip install deepgram-sdk)
  - ``WebSpeechSTT``     -- browser-native placeholder (pre-transcribed text)

Legacy compatibility: the module-level ``transcribe()``, ``check_stt_availability()``,
and ``get_stt_backend_name()`` functions are preserved for backwards compatibility
with the existing recorder-based voice input flow.
"""

from __future__ import annotations

import io
import logging
import os
import struct
from abc import ABC, abstractmethod
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# -- Audio constants (shared with recorder.py) --------------------------------

SAMPLE_RATE = 16000
CHANNELS = 1
BYTES_PER_SAMPLE = 2  # int16


# -- WAV helper ---------------------------------------------------------------

def _pcm_to_wav(pcm_bytes: bytes) -> bytes:
    """Wrap raw int16 PCM in a minimal WAV container."""
    byte_rate = SAMPLE_RATE * CHANNELS * BYTES_PER_SAMPLE
    block_align = CHANNELS * BYTES_PER_SAMPLE
    data_size = len(pcm_bytes)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM format
        CHANNELS,
        SAMPLE_RATE,
        byte_rate,
        block_align,
        16,  # bits per sample
        b"data",
        data_size,
    )
    return header + pcm_bytes


# =============================================================================
# Provider ABC
# =============================================================================


class STTProvider(ABC):
    """Abstract base class for speech-to-text providers."""

    @abstractmethod
    def transcribe(self, audio: bytes) -> str:
        """Transcribe raw audio bytes to text.

        Args:
            audio: Raw int16 PCM audio at 16 kHz mono, or provider-specific
                   format for cloud providers.

        Returns:
            Transcribed text string.
        """
        ...


# =============================================================================
# FasterWhisperSTT
# =============================================================================


class FasterWhisperSTT(STTProvider):
    """Local STT using faster-whisper.

    The model is lazy-loaded on first ``transcribe()`` call to avoid
    import-time overhead and to allow graceful degradation if the library
    is not installed.

    Parameters:
        model_size: Whisper model size ("tiny", "base", "small", "medium",
                    "large-v2", "large-v3").
        device: Compute device ("auto", "cpu", "cuda").
        language: BCP-47 language code or "auto" for detection.
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",
        language: str = "auto",
    ) -> None:
        self._model: Any = None
        self._model_size = model_size
        self._device = device
        self._language = language

    def _load_model(self) -> None:
        if self._model is not None:
            return

        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise RuntimeError(
                "faster-whisper is not installed. "
                "Install it with: pip install faster-whisper"
            )

        device = self._device
        if device == "auto":
            device = "cuda" if _has_cuda() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        self._model = WhisperModel(
            self._model_size,
            device=device,
            compute_type=compute_type,
        )
        logger.info(
            "FasterWhisperSTT loaded model=%s device=%s",
            self._model_size,
            device,
        )

    def transcribe(self, audio: bytes) -> str:
        """Transcribe raw int16 PCM audio (16 kHz, mono) to text."""
        if not audio:
            return ""

        self._load_model()

        import numpy as np

        audio_array = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        lang = None if self._language == "auto" else self._language

        segments, _info = self._model.transcribe(
            audio_array,
            language=lang,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
        )
        return " ".join(seg.text for seg in segments).strip()


# =============================================================================
# DeepgramSTT
# =============================================================================


class DeepgramSTT(STTProvider):
    """Cloud STT via Deepgram API.

    Parameters:
        api_key: Deepgram API key. Falls back to ``DEEPGRAM_API_KEY`` env var.
        language: BCP-47 language code.
        model: Deepgram model name (e.g. "nova-2").
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        language: str = "en",
        model: str = "nova-2",
    ) -> None:
        self._api_key = api_key or os.environ.get("DEEPGRAM_API_KEY", "")
        self._language = language
        self._model = model

    def transcribe(self, audio: bytes) -> str:
        """Transcribe audio bytes via Deepgram REST API."""
        if not audio:
            return ""

        if not self._api_key:
            raise RuntimeError(
                "Deepgram API key not configured. Set DEEPGRAM_API_KEY "
                "environment variable or pass api_key to DeepgramSTT."
            )

        try:
            from deepgram import DeepgramClient, PrerecordedOptions
        except ImportError:
            raise RuntimeError(
                "deepgram-sdk is not installed. "
                "Install it with: pip install deepgram-sdk"
            )

        wav_bytes = _pcm_to_wav(audio)
        client = DeepgramClient(self._api_key)
        payload = {"buffer": wav_bytes, "mimetype": "audio/wav"}
        options = PrerecordedOptions(
            model=self._model,
            language=self._language,
            smart_format=True,
        )
        response = client.listen.prerecorded.v("1").transcribe_file(
            payload, options
        )
        try:
            transcript = (
                response.results.channels[0].alternatives[0].transcript
            )
            return transcript.strip()
        except (AttributeError, IndexError):
            logger.warning("Deepgram returned empty transcript")
            return ""


# =============================================================================
# WebSpeechSTT
# =============================================================================


class WebSpeechSTT(STTProvider):
    """Placeholder for browser-native Web Speech API.

    In a web deployment, the browser performs STT natively and sends
    the transcript as UTF-8 text. This provider simply decodes the
    received bytes as text.
    """

    def transcribe(self, audio: bytes) -> str:
        """Decode pre-transcribed text from browser Web Speech API.

        Args:
            audio: UTF-8 encoded transcript text (not actual audio).
        """
        if isinstance(audio, bytes):
            return audio.decode("utf-8")
        return str(audio)


# =============================================================================
# Legacy compatibility functions
# =============================================================================

# Cached model handles for legacy API
_faster_whisper_model = None
_openai_whisper_model = None

DEFAULT_MODEL_SIZE = os.environ.get("SAIDO_AGENT_WHISPER_MODEL", "base")


def _has_cuda() -> bool:
    """Check if CUDA is available for GPU acceleration."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        pass
    try:
        import ctranslate2
        return "cuda" in ctranslate2.get_supported_compute_types("cuda")
    except Exception:
        return False


def check_stt_availability() -> tuple[bool, str | None]:
    """Return (available, reason_if_not).

    Legacy compatibility function. Checks for any available STT backend.
    """
    try:
        import faster_whisper  # noqa: F401
        return True, None
    except ImportError:
        pass
    try:
        import whisper  # noqa: F401
        return True, None
    except ImportError:
        pass
    if os.environ.get("OPENAI_API_KEY"):
        return True, None
    if os.environ.get("DEEPGRAM_API_KEY"):
        return True, None

    return False, (
        "No STT backend available.\n"
        "Install one of:\n"
        "  pip install faster-whisper   (local, recommended)\n"
        "  pip install openai-whisper   (local, original)\n"
        "  Set OPENAI_API_KEY to use the OpenAI Whisper cloud API\n"
        "  Set DEEPGRAM_API_KEY to use Deepgram cloud STT"
    )


def get_stt_backend_name() -> str:
    """Return a human-readable name of the backend that will be used."""
    try:
        import faster_whisper  # noqa: F401
        return f"faster-whisper ({DEFAULT_MODEL_SIZE})"
    except ImportError:
        pass
    try:
        import whisper  # noqa: F401
        return f"openai-whisper ({DEFAULT_MODEL_SIZE})"
    except ImportError:
        pass
    if os.environ.get("OPENAI_API_KEY"):
        return "OpenAI Whisper API"
    if os.environ.get("DEEPGRAM_API_KEY"):
        return "Deepgram API"
    return "(none)"


def _keyterms_to_prompt(keyterms: List[str]) -> str:
    """Convert a list of keywords into a Whisper initial_prompt string."""
    if not keyterms:
        return ""
    return ", ".join(keyterms[:40])


def transcribe(
    pcm_bytes: bytes,
    keyterms: Optional[List[str]] = None,
    language: str = "auto",
) -> str:
    """Transcribe raw PCM audio to text (legacy compatibility function).

    Args:
        pcm_bytes: Raw int16 PCM, 16 kHz, mono.
        keyterms: Coding-domain vocabulary hints.
        language: BCP-47 language code, or 'auto' for detection.

    Returns:
        Transcribed text, or empty string if audio contains no speech.
    """
    if not pcm_bytes:
        return ""

    terms = keyterms or []
    lang = None if language == "auto" else language

    # faster-whisper (local, preferred)
    try:
        import faster_whisper  # noqa: F401

        global _faster_whisper_model
        if _faster_whisper_model is None:
            from faster_whisper import WhisperModel

            device = "cuda" if _has_cuda() else "cpu"
            compute = "float16" if device == "cuda" else "int8"
            _faster_whisper_model = WhisperModel(
                DEFAULT_MODEL_SIZE, device=device, compute_type=compute
            )

        import numpy as np

        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        initial_prompt = _keyterms_to_prompt(terms)
        segments, _info = _faster_whisper_model.transcribe(
            audio,
            language=lang,
            initial_prompt=initial_prompt,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
        )
        return " ".join(seg.text for seg in segments).strip()
    except ImportError:
        pass

    # openai-whisper (local, fallback)
    try:
        import whisper  # noqa: F401
        import numpy as np

        global _openai_whisper_model
        if _openai_whisper_model is None:
            _openai_whisper_model = whisper.load_model(DEFAULT_MODEL_SIZE)

        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        initial_prompt = _keyterms_to_prompt(terms)
        options: dict = {"initial_prompt": initial_prompt} if initial_prompt else {}
        if lang:
            options["language"] = lang
        result = _openai_whisper_model.transcribe(audio, **options)
        return result.get("text", "").strip()
    except ImportError:
        pass

    # OpenAI Whisper API (cloud, last resort)
    if os.environ.get("OPENAI_API_KEY"):
        from openai import OpenAI

        client = OpenAI()
        wav = _pcm_to_wav(pcm_bytes)
        kwargs: dict = {
            "model": "whisper-1",
            "file": ("audio.wav", io.BytesIO(wav), "audio/wav"),
        }
        if lang:
            kwargs["language"] = lang
        transcript = client.audio.transcriptions.create(**kwargs)
        return transcript.text.strip()

    raise RuntimeError(
        "No STT backend available.\n"
        "Install faster-whisper:  pip install faster-whisper\n"
        "Or set OPENAI_API_KEY to use the OpenAI Whisper cloud API."
    )
