"""Voice pipeline configuration for Saido Agent.

Defines the ``VoiceConfig`` dataclass used to configure STT, TTS,
activation mode, and other voice pipeline parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VoiceConfig:
    """Configuration for the voice pipeline.

    Attributes:
        stt: STT provider name. One of "faster-whisper", "deepgram",
             "web-speech".
        tts: TTS provider name. One of "kokoro", "voxtral", "elevenlabs",
             "openai".
        activation: Activation mode. One of "push-to-talk", "voice-activity",
                    "wake-word".
        voice_style: Voice persona style hint passed to TTS providers.
        max_response_tokens: Maximum tokens for agent response (shorter
                             responses keep voice latency low).
        edge_mode: When True, prefer smaller models and skip optional
                   processing stages for lower latency on edge devices.
        stt_model_size: Whisper model size for local STT backends.
        stt_api_key: API key for cloud STT providers (e.g. Deepgram).
        tts_api_key: API key for cloud TTS providers (e.g. ElevenLabs).
        tts_voice_id: Voice identifier for cloud TTS providers.
        vad_threshold: VAD speech probability threshold (0.0-1.0).
        vad_min_speech_ms: Minimum speech segment duration in ms.
        vad_min_silence_ms: Minimum silence duration to split segments in ms.
        sample_rate: Audio sample rate in Hz.
    """

    stt: str = "faster-whisper"
    tts: str = "kokoro"
    activation: str = "push-to-talk"
    voice_style: str = "friendly"
    max_response_tokens: int = 150
    edge_mode: bool = False
    stt_model_size: str = "small"
    stt_api_key: Optional[str] = None
    tts_api_key: Optional[str] = None
    tts_voice_id: Optional[str] = None
    vad_threshold: float = 0.5
    vad_min_speech_ms: int = 250
    vad_min_silence_ms: int = 300
    sample_rate: int = 16000

    # Valid choices for validation
    VALID_STT_PROVIDERS: tuple[str, ...] = (
        "faster-whisper",
        "deepgram",
        "web-speech",
    )
    VALID_TTS_PROVIDERS: tuple[str, ...] = (
        "kokoro",
        "voxtral",
        "elevenlabs",
        "openai",
    )
    VALID_ACTIVATIONS: tuple[str, ...] = (
        "push-to-talk",
        "voice-activity",
        "wake-word",
    )

    def validate(self) -> list[str]:
        """Validate configuration and return a list of error messages.

        Returns an empty list if the configuration is valid.
        """
        errors: list[str] = []

        if self.stt not in self.VALID_STT_PROVIDERS:
            errors.append(
                f"Invalid STT provider '{self.stt}'. "
                f"Valid options: {', '.join(self.VALID_STT_PROVIDERS)}"
            )
        if self.tts not in self.VALID_TTS_PROVIDERS:
            errors.append(
                f"Invalid TTS provider '{self.tts}'. "
                f"Valid options: {', '.join(self.VALID_TTS_PROVIDERS)}"
            )
        if self.activation not in self.VALID_ACTIVATIONS:
            errors.append(
                f"Invalid activation mode '{self.activation}'. "
                f"Valid options: {', '.join(self.VALID_ACTIVATIONS)}"
            )
        if not 0.0 <= self.vad_threshold <= 1.0:
            errors.append(
                f"VAD threshold must be between 0.0 and 1.0, got {self.vad_threshold}"
            )
        if self.max_response_tokens < 1:
            errors.append(
                f"max_response_tokens must be >= 1, got {self.max_response_tokens}"
            )

        return errors
