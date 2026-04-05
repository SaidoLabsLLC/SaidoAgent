"""Voice pipeline orchestrator for Saido Agent.

Wires together STT, VAD, TTS, and the SaidoAgent query engine into a
single ``VoicePipeline`` that accepts audio input and returns audio output.

Pipeline flow:
  1. Audio -> VAD (detect speech boundaries) -> STT (transcribe)
  2. Transcript -> agent.query() with knowledge grounding
  3. Answer -> TTS synthesize (sentence-by-sentence for streaming)
  4. Return audio bytes + metadata (latency measurements)
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

from saido_agent.voice.config import VoiceConfig
from saido_agent.voice.stt import (
    DeepgramSTT,
    FasterWhisperSTT,
    STTProvider,
    WebSpeechSTT,
)
from saido_agent.voice.tts import (
    ElevenLabsTTS,
    KokoroTTS,
    OpenAITTS,
    TTSProvider,
    VoxtralTTS,
)
from saido_agent.voice.vad import SileroVAD

logger = logging.getLogger(__name__)


# =============================================================================
# Result dataclass
# =============================================================================


@dataclass
class VoiceResult:
    """Result of a voice pipeline processing cycle.

    Attributes:
        transcript: STT output -- what the user said.
        answer: Agent's text response.
        audio_out: Synthesized audio bytes of the agent's response.
        citations: Knowledge citations from the agent query.
        latency_ms: Per-stage latency measurements in milliseconds.
    """

    transcript: str
    answer: str
    audio_out: bytes
    citations: list = field(default_factory=list)
    latency_ms: dict = field(default_factory=dict)


# =============================================================================
# Sentence splitter for streaming TTS
# =============================================================================

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences for incremental TTS synthesis."""
    sentences = _SENTENCE_RE.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


# =============================================================================
# VoicePipeline
# =============================================================================


class VoicePipeline:
    """Bidirectional voice pipeline: STT -> Agent -> TTS.

    Accepts raw audio bytes and returns synthesized audio along with
    the full text transcript and agent response.

    Parameters:
        agent: A ``SaidoAgent`` instance (or any object with a ``.query()``
               method that returns an object with ``.answer`` and
               ``.citations`` attributes).
        stt: STT provider name or ``STTProvider`` instance.
        tts: TTS provider name or ``TTSProvider`` instance.
        activation: Activation mode ("push-to-talk", "voice-activity",
                    "wake-word").
        voice_style: Voice persona hint for TTS.
        max_response_tokens: Max tokens for agent response.
        edge_mode: When True, skip VAD and use smaller models.
        config: Optional ``VoiceConfig`` (overrides individual params).
    """

    def __init__(
        self,
        agent: Any,
        stt: str | STTProvider = "faster-whisper",
        tts: str | TTSProvider = "kokoro",
        activation: str = "push-to-talk",
        voice_style: str = "friendly",
        max_response_tokens: int = 150,
        edge_mode: bool = False,
        config: Optional[VoiceConfig] = None,
    ) -> None:
        self._agent = agent

        # If a VoiceConfig is provided, it takes precedence
        if config is not None:
            stt = config.stt
            tts = config.tts
            activation = config.activation
            voice_style = config.voice_style
            max_response_tokens = config.max_response_tokens
            edge_mode = config.edge_mode

        self._activation = activation
        self._voice_style = voice_style
        self._max_response_tokens = max_response_tokens
        self._edge_mode = edge_mode

        # Initialize providers
        self._stt = self._init_stt(stt, config)
        self._tts = self._init_tts(tts, config)
        self._vad = SileroVAD(
            threshold=config.vad_threshold if config else 0.5,
            min_speech_ms=config.vad_min_speech_ms if config else 250,
            min_silence_ms=config.vad_min_silence_ms if config else 300,
        )

        logger.info(
            "VoicePipeline initialized (stt=%s, tts=%s, activation=%s, edge=%s)",
            type(self._stt).__name__,
            type(self._tts).__name__,
            self._activation,
            self._edge_mode,
        )

    # -- Provider initialization ----------------------------------------------

    def _init_stt(
        self,
        stt: str | STTProvider,
        config: Optional[VoiceConfig],
    ) -> STTProvider:
        """Resolve an STT provider from a name string or instance."""
        if isinstance(stt, STTProvider):
            return stt

        stt_name = stt.lower().strip()
        model_size = config.stt_model_size if config else "small"

        if stt_name == "faster-whisper":
            return FasterWhisperSTT(model_size=model_size)
        elif stt_name == "deepgram":
            api_key = config.stt_api_key if config else None
            return DeepgramSTT(api_key=api_key)
        elif stt_name == "web-speech":
            return WebSpeechSTT()
        else:
            logger.warning(
                "Unknown STT provider '%s', falling back to FasterWhisperSTT",
                stt_name,
            )
            return FasterWhisperSTT(model_size=model_size)

    def _init_tts(
        self,
        tts: str | TTSProvider,
        config: Optional[VoiceConfig],
    ) -> TTSProvider:
        """Resolve a TTS provider from a name string or instance."""
        if isinstance(tts, TTSProvider):
            return tts

        tts_name = tts.lower().strip()
        api_key = config.tts_api_key if config else None
        voice_id = config.tts_voice_id if config else None

        if tts_name == "kokoro":
            return KokoroTTS()
        elif tts_name == "voxtral":
            return VoxtralTTS()
        elif tts_name == "elevenlabs":
            kwargs: dict[str, Any] = {}
            if api_key:
                kwargs["api_key"] = api_key
            if voice_id:
                kwargs["voice_id"] = voice_id
            return ElevenLabsTTS(**kwargs)
        elif tts_name == "openai":
            kwargs = {}
            if api_key:
                kwargs["api_key"] = api_key
            return OpenAITTS(**kwargs)
        else:
            logger.warning(
                "Unknown TTS provider '%s', falling back to KokoroTTS",
                tts_name,
            )
            return KokoroTTS()

    # -- Properties -----------------------------------------------------------

    @property
    def stt(self) -> STTProvider:
        """The active STT provider."""
        return self._stt

    @property
    def tts(self) -> TTSProvider:
        """The active TTS provider."""
        return self._tts

    @property
    def vad(self) -> SileroVAD:
        """The VAD instance."""
        return self._vad

    @property
    def activation(self) -> str:
        """The current activation mode."""
        return self._activation

    # -- Main processing methods ----------------------------------------------

    async def process(
        self,
        audio_bytes: bytes,
        context: Optional[dict] = None,
    ) -> VoiceResult:
        """Process voice input through the full pipeline.

        1. VAD: detect speech boundaries (skip in edge mode)
        2. STT: transcribe speech to text
        3. Agent: generate knowledge-grounded response
        4. TTS: synthesize response to audio

        Args:
            audio_bytes: Raw int16 PCM audio at 16 kHz mono.
            context: Optional context dict passed to agent.query().

        Returns:
            A ``VoiceResult`` with transcript, answer, audio, and latency.
        """
        latency: dict[str, float] = {}
        t_start = time.perf_counter()

        # --- Stage 1: VAD ---
        t0 = time.perf_counter()
        if self._edge_mode:
            # Skip VAD in edge mode to reduce latency
            speech_audio = audio_bytes
        else:
            segments = self._vad.detect_speech(audio_bytes)
            speech_audio = b"".join(segments) if segments else audio_bytes
        latency["vad_ms"] = (time.perf_counter() - t0) * 1000

        # --- Stage 2: STT ---
        t0 = time.perf_counter()
        transcript = self._stt.transcribe(speech_audio)
        latency["stt_ms"] = (time.perf_counter() - t0) * 1000

        if not transcript.strip():
            latency["total_ms"] = (time.perf_counter() - t_start) * 1000
            return VoiceResult(
                transcript="",
                answer="",
                audio_out=b"",
                latency_ms=latency,
            )

        # --- Stage 3: Agent query ---
        t0 = time.perf_counter()
        query_result = self._agent.query(transcript, context=context)
        answer = query_result.answer
        citations = []
        if hasattr(query_result, "citations"):
            citations = [
                {"slug": c.slug, "title": c.title}
                for c in query_result.citations
                if hasattr(c, "slug")
            ]
        latency["llm_ms"] = (time.perf_counter() - t0) * 1000

        # --- Stage 4: TTS ---
        t0 = time.perf_counter()
        audio_out = self._tts.synthesize(answer)
        latency["tts_ms"] = (time.perf_counter() - t0) * 1000

        latency["total_ms"] = (time.perf_counter() - t_start) * 1000

        return VoiceResult(
            transcript=transcript,
            answer=answer,
            audio_out=audio_out,
            citations=citations,
            latency_ms=latency,
        )

    async def stream(
        self,
        audio_bytes: bytes,
        context: Optional[dict] = None,
    ) -> AsyncIterator[bytes]:
        """Stream audio response chunks for real-time playback.

        Performs STT and agent query, then streams TTS output
        sentence-by-sentence to minimize time-to-first-audio.

        Args:
            audio_bytes: Raw int16 PCM audio at 16 kHz mono.
            context: Optional context dict passed to agent.query().

        Yields:
            Audio byte chunks for sequential playback.
        """
        # VAD + STT
        if self._edge_mode:
            speech_audio = audio_bytes
        else:
            segments = self._vad.detect_speech(audio_bytes)
            speech_audio = b"".join(segments) if segments else audio_bytes

        transcript = self._stt.transcribe(speech_audio)
        if not transcript.strip():
            return

        # Agent query
        query_result = self._agent.query(transcript, context=context)
        answer = query_result.answer

        if not answer.strip():
            return

        # Stream TTS sentence by sentence
        sentences = _split_sentences(answer)
        if not sentences:
            sentences = [answer]

        for sentence in sentences:
            async for chunk in self._tts.stream(sentence):
                yield chunk
