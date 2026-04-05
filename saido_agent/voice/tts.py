"""Text-to-speech (TTS) provider abstraction for Saido Agent voice pipeline.

Provides a provider-based TTS architecture with lazy model loading:

  - ``KokoroTTS``     -- lightweight local TTS (82M params, default)
  - ``VoxtralTTS``    -- higher quality local TTS (placeholder)
  - ``ElevenLabsTTS`` -- cloud premium TTS via ElevenLabs API
  - ``OpenAITTS``     -- cloud TTS via OpenAI API

All providers implement ``synthesize()`` for full audio generation and
``stream()`` for incremental chunk-based streaming.
"""

from __future__ import annotations

import io
import logging
import os
import struct
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Optional

logger = logging.getLogger(__name__)

# Audio format constants
SAMPLE_RATE = 24000  # TTS typically outputs at 24kHz
CHANNELS = 1
BYTES_PER_SAMPLE = 2  # int16


def _generate_silence(duration_ms: int = 100, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Generate silent PCM audio of the given duration."""
    num_samples = int(sample_rate * duration_ms / 1000)
    return b"\x00\x00" * num_samples


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Wrap raw int16 PCM in a WAV container."""
    byte_rate = sample_rate * CHANNELS * BYTES_PER_SAMPLE
    block_align = CHANNELS * BYTES_PER_SAMPLE
    data_size = len(pcm_bytes)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        CHANNELS,
        sample_rate,
        byte_rate,
        block_align,
        16,
        b"data",
        data_size,
    )
    return header + pcm_bytes


# =============================================================================
# Provider ABC
# =============================================================================


class TTSProvider(ABC):
    """Abstract base class for text-to-speech providers."""

    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        """Synthesize text to audio bytes.

        Args:
            text: Text to synthesize into speech.

        Returns:
            Raw audio bytes (format is provider-specific, typically WAV or
            raw PCM int16).
        """
        ...

    @abstractmethod
    def stream(self, text: str) -> AsyncIterator[bytes]:
        """Yield audio chunks for streaming playback.

        Implementations should be async generators (``async def`` with
        ``yield``).  The return type is ``AsyncIterator[bytes]`` so that
        both coroutine-based and async-generator-based implementations
        are accepted.

        Args:
            text: Text to synthesize.

        Yields:
            Audio byte chunks suitable for sequential playback.
        """
        ...


# =============================================================================
# KokoroTTS -- Lightweight local TTS (default)
# =============================================================================


class KokoroTTS(TTSProvider):
    """Lightweight fallback TTS (82M params).

    Uses the Kokoro TTS model for fast local synthesis. The model is
    lazy-loaded on first use.

    Parameters:
        voice: Voice preset name.
        speed: Speech speed multiplier (0.5-2.0).
    """

    def __init__(self, voice: str = "af_heart", speed: float = 1.0) -> None:
        self._voice = voice
        self._speed = speed
        self._model: Any = None
        self._available: bool | None = None

    def _load_model(self) -> None:
        if self._model is not None:
            return

        try:
            from kokoro import KPipeline
            self._model = KPipeline(lang_code="a")
            self._available = True
            logger.info("KokoroTTS model loaded (voice=%s)", self._voice)
        except ImportError:
            self._available = False
            raise RuntimeError(
                "kokoro is not installed. "
                "Install it with: pip install kokoro"
            )
        except Exception as exc:
            self._available = False
            raise RuntimeError(f"Failed to load Kokoro TTS model: {exc}")

    def synthesize(self, text: str) -> bytes:
        """Synthesize text to WAV audio bytes."""
        if not text.strip():
            return _generate_silence()

        self._load_model()

        try:
            generator = self._model(
                text, voice=self._voice, speed=self._speed
            )
            # Collect all audio chunks
            audio_chunks = []
            for _gs, _ps, audio in generator:
                audio_chunks.append(audio)

            if not audio_chunks:
                return _generate_silence()

            import numpy as np
            combined = np.concatenate(audio_chunks)
            pcm = (combined * 32767).astype(np.int16).tobytes()
            return _pcm_to_wav(pcm)

        except Exception as exc:
            logger.error("KokoroTTS synthesis failed: %s", exc)
            return _generate_silence()

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        """Stream audio chunks sentence by sentence."""
        if not text.strip():
            yield _generate_silence()
            return

        self._load_model()

        try:
            generator = self._model(
                text, voice=self._voice, speed=self._speed
            )
            for _gs, _ps, audio in generator:
                import numpy as np
                pcm = (audio * 32767).astype(np.int16).tobytes()
                yield _pcm_to_wav(pcm)
        except Exception as exc:
            logger.error("KokoroTTS streaming failed: %s", exc)
            yield _generate_silence()


# =============================================================================
# VoxtralTTS -- Higher quality local TTS (placeholder)
# =============================================================================


class VoxtralTTS(TTSProvider):
    """Default local TTS using Voxtral model.

    This is a placeholder implementation. The actual Voxtral model loading
    will be implemented when the model becomes available. Currently falls
    back to generating silence with a warning.

    Parameters:
        voice: Voice preset name.
    """

    def __init__(self, voice: str = "default") -> None:
        self._voice = voice
        self._model: Any = None

    def synthesize(self, text: str) -> bytes:
        """Synthesize text to audio bytes (placeholder)."""
        logger.warning(
            "VoxtralTTS is a placeholder -- returning silence. "
            "Use KokoroTTS or a cloud provider for actual synthesis."
        )
        # Return silence proportional to text length
        duration_ms = max(100, len(text) * 50)
        return _pcm_to_wav(_generate_silence(duration_ms))

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        """Stream audio chunks (placeholder)."""
        yield self.synthesize(text)


# =============================================================================
# ElevenLabsTTS -- Cloud premium TTS
# =============================================================================


class ElevenLabsTTS(TTSProvider):
    """Cloud premium TTS via ElevenLabs API.

    Parameters:
        api_key: ElevenLabs API key. Falls back to ``ELEVENLABS_API_KEY`` env.
        voice_id: ElevenLabs voice ID.
        model_id: ElevenLabs model ID.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",  # Rachel
        model_id: str = "eleven_monolingual_v1",
    ) -> None:
        self._api_key = api_key or os.environ.get("ELEVENLABS_API_KEY", "")
        self._voice_id = voice_id
        self._model_id = model_id

    def _ensure_api_key(self) -> None:
        if not self._api_key:
            raise RuntimeError(
                "ElevenLabs API key not configured. Set ELEVENLABS_API_KEY "
                "environment variable or pass api_key to ElevenLabsTTS."
            )

    def synthesize(self, text: str) -> bytes:
        """Synthesize text via ElevenLabs API."""
        if not text.strip():
            return _generate_silence()

        self._ensure_api_key()

        try:
            from elevenlabs import ElevenLabs as ElevenLabsClient
        except ImportError:
            raise RuntimeError(
                "elevenlabs is not installed. "
                "Install it with: pip install elevenlabs"
            )

        client = ElevenLabsClient(api_key=self._api_key)
        audio_generator = client.text_to_speech.convert(
            voice_id=self._voice_id,
            text=text,
            model_id=self._model_id,
            output_format="mp3_44100_128",
        )

        # Collect all chunks
        audio_data = b"".join(audio_generator)
        return audio_data

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        """Stream audio chunks from ElevenLabs API."""
        if not text.strip():
            yield _generate_silence()
            return

        self._ensure_api_key()

        try:
            from elevenlabs import ElevenLabs as ElevenLabsClient
        except ImportError:
            raise RuntimeError(
                "elevenlabs is not installed. "
                "Install it with: pip install elevenlabs"
            )

        client = ElevenLabsClient(api_key=self._api_key)
        audio_generator = client.text_to_speech.convert(
            voice_id=self._voice_id,
            text=text,
            model_id=self._model_id,
            output_format="mp3_44100_128",
        )

        for chunk in audio_generator:
            if chunk:
                yield chunk


# =============================================================================
# OpenAITTS -- Cloud TTS via OpenAI API
# =============================================================================


class OpenAITTS(TTSProvider):
    """Cloud TTS via OpenAI API.

    Parameters:
        api_key: OpenAI API key. Falls back to ``OPENAI_API_KEY`` env var.
        voice: OpenAI voice name (alloy, echo, fable, onyx, nova, shimmer).
        model: OpenAI TTS model (tts-1, tts-1-hd).
        speed: Speech speed multiplier (0.25-4.0).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice: str = "alloy",
        model: str = "tts-1",
        speed: float = 1.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._voice = voice
        self._model = model
        self._speed = speed

    def _ensure_api_key(self) -> None:
        if not self._api_key:
            raise RuntimeError(
                "OpenAI API key not configured. Set OPENAI_API_KEY "
                "environment variable or pass api_key to OpenAITTS."
            )

    def synthesize(self, text: str) -> bytes:
        """Synthesize text via OpenAI TTS API."""
        if not text.strip():
            return _generate_silence()

        self._ensure_api_key()

        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError(
                "openai is not installed. "
                "Install it with: pip install openai"
            )

        client = OpenAI(api_key=self._api_key)
        response = client.audio.speech.create(
            model=self._model,
            voice=self._voice,
            input=text,
            speed=self._speed,
            response_format="wav",
        )
        return response.content

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        """Stream audio chunks from OpenAI TTS API."""
        if not text.strip():
            yield _generate_silence()
            return

        self._ensure_api_key()

        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError(
                "openai is not installed. Install it with: pip install openai"
            )

        client = OpenAI(api_key=self._api_key)
        response = client.audio.speech.create(
            model=self._model,
            voice=self._voice,
            input=text,
            speed=self._speed,
            response_format="wav",
        )
        # OpenAI returns full audio; split into chunks for streaming
        content = response.content
        chunk_size = 4096
        for i in range(0, len(content), chunk_size):
            yield content[i : i + chunk_size]
