"""Voice Activity Detection (VAD) for Saido Agent voice pipeline.

Uses Silero VAD for high-accuracy speech boundary detection.
The model is lazy-loaded on first use to avoid import-time overhead.

If ``torch`` or the Silero model is not available, a simple energy-based
fallback is used so the pipeline can still function (with reduced accuracy).
"""

from __future__ import annotations

import logging
import struct
from typing import Any, Optional

logger = logging.getLogger(__name__)


class SileroVAD:
    """Voice Activity Detection using Silero VAD model.

    Parameters:
        threshold: Speech probability threshold (0.0-1.0). Chunks with
            probability above this are classified as speech.
        min_speech_ms: Minimum speech segment duration in milliseconds.
            Segments shorter than this are discarded.
        min_silence_ms: Minimum silence duration in milliseconds to split
            speech segments.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        min_silence_ms: int = 300,
    ) -> None:
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self._model: Any = None
        self._torch: Any = None
        self._available: Optional[bool] = None

    @property
    def available(self) -> bool:
        """Check if Silero VAD model can be loaded."""
        if self._available is None:
            try:
                import torch  # noqa: F401
                self._available = True
            except ImportError:
                self._available = False
                logger.warning(
                    "torch not installed -- VAD will use energy-based fallback. "
                    "Install torch for better accuracy: pip install torch"
                )
        return self._available

    def _load_model(self) -> None:
        """Lazy-load the Silero VAD model."""
        if self._model is not None:
            return

        if not self.available:
            return

        try:
            import torch

            self._torch = torch
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            self._model = model
            logger.info("Silero VAD model loaded successfully")
        except Exception as exc:
            logger.warning("Failed to load Silero VAD model: %s", exc)
            self._available = False

    def is_speech(self, audio_chunk: bytes, sample_rate: int = 16000) -> bool:
        """Check if a single audio chunk contains speech.

        Args:
            audio_chunk: Raw int16 PCM audio bytes.
            sample_rate: Audio sample rate in Hz.

        Returns:
            True if speech is detected in the chunk.
        """
        if not audio_chunk:
            return False

        self._load_model()

        if self._model is not None and self._torch is not None:
            return self._is_speech_silero(audio_chunk, sample_rate)

        return self._is_speech_energy(audio_chunk)

    def _is_speech_silero(self, audio_chunk: bytes, sample_rate: int) -> bool:
        """Speech detection using Silero VAD model."""
        import torch

        # Convert int16 PCM to float32 tensor
        num_samples = len(audio_chunk) // 2
        samples = struct.unpack(f"<{num_samples}h", audio_chunk)
        audio_tensor = torch.FloatTensor(samples) / 32768.0

        # Silero VAD expects specific chunk sizes (512 for 16kHz)
        speech_prob = self._model(audio_tensor, sample_rate).item()
        return speech_prob >= self.threshold

    def _is_speech_energy(self, audio_chunk: bytes) -> bool:
        """Simple energy-based speech detection fallback."""
        if len(audio_chunk) < 2:
            return False

        num_samples = len(audio_chunk) // 2
        samples = struct.unpack(f"<{num_samples}h", audio_chunk)

        # Calculate RMS energy
        sum_sq = sum(s * s for s in samples)
        rms = (sum_sq / num_samples) ** 0.5
        # Normalize to 0-1 range (int16 max = 32768)
        normalized_rms = rms / 32768.0

        # Energy threshold -- roughly equivalent to Silero's 0.5 threshold
        energy_threshold = 0.015
        return normalized_rms >= energy_threshold

    def detect_speech(
        self,
        audio_bytes: bytes,
        sample_rate: int = 16000,
    ) -> list[bytes]:
        """Split audio into speech segments.

        Uses VAD to identify speech regions and returns only the audio
        segments that contain speech, filtering out silence.

        Args:
            audio_bytes: Raw int16 PCM audio bytes.
            sample_rate: Audio sample rate in Hz.

        Returns:
            List of audio byte segments containing speech.
        """
        if not audio_bytes:
            return []

        self._load_model()

        bytes_per_sample = 2  # int16
        # Process in chunks of ~30ms (480 samples at 16kHz)
        chunk_samples = int(sample_rate * 0.03)
        chunk_bytes = chunk_samples * bytes_per_sample

        min_speech_samples = int(sample_rate * self.min_speech_ms / 1000)
        min_silence_samples = int(sample_rate * self.min_silence_ms / 1000)
        min_speech_bytes = min_speech_samples * bytes_per_sample
        min_silence_chunks = max(1, min_silence_samples // chunk_samples)

        segments: list[bytes] = []
        current_segment = bytearray()
        silence_count = 0
        in_speech = False

        offset = 0
        while offset + chunk_bytes <= len(audio_bytes):
            chunk = audio_bytes[offset : offset + chunk_bytes]
            speech = self.is_speech(chunk, sample_rate)

            if speech:
                if not in_speech:
                    in_speech = True
                    silence_count = 0
                current_segment.extend(chunk)
                silence_count = 0
            else:
                if in_speech:
                    silence_count += 1
                    current_segment.extend(chunk)  # Include silence in segment

                    if silence_count >= min_silence_chunks:
                        # End of speech segment
                        if len(current_segment) >= min_speech_bytes:
                            segments.append(bytes(current_segment))
                        current_segment = bytearray()
                        in_speech = False
                        silence_count = 0

            offset += chunk_bytes

        # Handle remaining audio
        if in_speech and len(current_segment) >= min_speech_bytes:
            segments.append(bytes(current_segment))

        return segments
