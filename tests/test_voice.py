"""Tests for saido_agent.voice -- Voice pipeline SDK (Phase 3).

Tests cover:
  - STT providers transcribe correctly (mocked audio)
  - VAD detects speech segments (mocked model)
  - TTS providers synthesize text to bytes (mocked)
  - VoicePipeline orchestrates the full flow
  - Streaming produces chunks incrementally
  - Latency metrics captured in pipeline results
  - Activation modes configured correctly
  - Edge mode optimizations
  - Graceful degradation when libraries not installed
  - VoiceConfig validation
"""

from __future__ import annotations

import asyncio
import struct
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from saido_agent.voice.config import VoiceConfig
from saido_agent.voice.pipeline import VoicePipeline, VoiceResult, _split_sentences
from saido_agent.voice.stt import (
    DeepgramSTT,
    FasterWhisperSTT,
    STTProvider,
    WebSpeechSTT,
)
from saido_agent.voice.tts import (
    EdgeTTS,
    ElevenLabsTTS,
    KokoroTTS,
    OpenAITTS,
    PiperTTS,
    TTSProvider,
    VoxtralTTS,
    _generate_silence,
    detect_best_tts,
)
from saido_agent.voice.vad import SileroVAD


# =============================================================================
# Helpers
# =============================================================================


def _make_pcm_audio(duration_ms: int = 500, sample_rate: int = 16000) -> bytes:
    """Generate synthetic PCM audio (sine wave) for testing."""
    import math

    num_samples = int(sample_rate * duration_ms / 1000)
    freq = 440  # Hz
    samples = []
    for i in range(num_samples):
        t = i / sample_rate
        value = int(16000 * math.sin(2 * math.pi * freq * t))
        samples.append(value)
    return struct.pack(f"<{num_samples}h", *samples)


def _make_silent_audio(duration_ms: int = 500, sample_rate: int = 16000) -> bytes:
    """Generate silent PCM audio for testing."""
    num_samples = int(sample_rate * duration_ms / 1000)
    return b"\x00\x00" * num_samples


@dataclass
class MockCitation:
    slug: str = "test-doc"
    title: str = "Test Document"
    excerpt: str = "test excerpt"
    verified: bool = True


@dataclass
class MockQueryResult:
    answer: str = "This is a test answer."
    citations: list = field(default_factory=lambda: [MockCitation()])
    confidence: str = "high"
    retrieval_stats: dict = field(default_factory=dict)
    tokens_used: int = 42
    provider: str = "mock"


class MockAgent:
    """Mock SaidoAgent for pipeline testing."""

    def __init__(self, answer: str = "This is a test answer."):
        self._answer = answer
        self._voice_config = None
        self._voice_pipeline = None

    def query(self, question: str, context: Optional[dict] = None) -> MockQueryResult:
        return MockQueryResult(answer=self._answer)


class MockSTT(STTProvider):
    """Mock STT that returns a fixed transcript."""

    def __init__(self, transcript: str = "hello world"):
        self._transcript = transcript

    def transcribe(self, audio: bytes) -> str:
        return self._transcript if audio else ""


class MockTTS(TTSProvider):
    """Mock TTS that returns fixed audio bytes."""

    def __init__(self, audio: bytes = b"FAKE_AUDIO_DATA"):
        self._audio = audio

    def synthesize(self, text: str) -> bytes:
        return self._audio if text.strip() else b""

    async def stream(self, text: str):
        if text.strip():
            # Split into 2 chunks
            mid = len(self._audio) // 2
            yield self._audio[:mid]
            yield self._audio[mid:]


# =============================================================================
# VoiceConfig tests
# =============================================================================


class TestVoiceConfig:
    def test_default_config(self):
        config = VoiceConfig()
        assert config.stt == "faster-whisper"
        assert config.tts == "kokoro"
        assert config.activation == "push-to-talk"
        assert config.voice_style == "friendly"
        assert config.max_response_tokens == 150
        assert config.edge_mode is False
        assert config.stt_model_size == "small"
        assert config.sample_rate == 16000

    def test_custom_config(self):
        config = VoiceConfig(
            stt="deepgram",
            tts="elevenlabs",
            activation="voice-activity",
            edge_mode=True,
            max_response_tokens=50,
        )
        assert config.stt == "deepgram"
        assert config.tts == "elevenlabs"
        assert config.activation == "voice-activity"
        assert config.edge_mode is True
        assert config.max_response_tokens == 50

    def test_new_tts_providers_valid(self):
        """Piper, edge-tts, and auto should be valid TTS provider names."""
        for provider in ("piper", "edge-tts", "auto"):
            config = VoiceConfig(tts=provider)
            errors = config.validate()
            assert errors == [], f"Expected no errors for tts='{provider}', got {errors}"

    def test_validate_valid_config(self):
        config = VoiceConfig()
        errors = config.validate()
        assert errors == []

    def test_validate_invalid_stt(self):
        config = VoiceConfig(stt="invalid-provider")
        errors = config.validate()
        assert len(errors) == 1
        assert "Invalid STT provider" in errors[0]

    def test_validate_invalid_tts(self):
        config = VoiceConfig(tts="invalid-provider")
        errors = config.validate()
        assert len(errors) == 1
        assert "Invalid TTS provider" in errors[0]

    def test_validate_invalid_activation(self):
        config = VoiceConfig(activation="invalid-mode")
        errors = config.validate()
        assert len(errors) == 1
        assert "Invalid activation mode" in errors[0]

    def test_validate_invalid_vad_threshold(self):
        config = VoiceConfig(vad_threshold=1.5)
        errors = config.validate()
        assert len(errors) == 1
        assert "VAD threshold" in errors[0]

    def test_validate_invalid_max_tokens(self):
        config = VoiceConfig(max_response_tokens=0)
        errors = config.validate()
        assert len(errors) == 1
        assert "max_response_tokens" in errors[0]

    def test_validate_multiple_errors(self):
        config = VoiceConfig(
            stt="bad", tts="bad", activation="bad",
            vad_threshold=-1, max_response_tokens=-5,
        )
        errors = config.validate()
        assert len(errors) == 5


# =============================================================================
# STT provider tests
# =============================================================================


class TestWebSpeechSTT:
    def test_transcribe_bytes(self):
        stt = WebSpeechSTT()
        result = stt.transcribe(b"hello world")
        assert result == "hello world"

    def test_transcribe_utf8(self):
        stt = WebSpeechSTT()
        result = stt.transcribe("test input".encode("utf-8"))
        assert result == "test input"

    def test_transcribe_empty(self):
        stt = WebSpeechSTT()
        result = stt.transcribe(b"")
        assert result == ""


class TestFasterWhisperSTT:
    def test_graceful_error_when_not_installed(self):
        stt = FasterWhisperSTT()
        with patch.dict("sys.modules", {"faster_whisper": None}):
            with pytest.raises(RuntimeError, match="faster-whisper is not installed"):
                stt.transcribe(b"audio")

    def test_transcribe_empty_returns_empty(self):
        stt = FasterWhisperSTT()
        assert stt.transcribe(b"") == ""

    def test_model_lazy_loaded(self):
        """Model should not be loaded at construction time."""
        stt = FasterWhisperSTT()
        assert stt._model is None


class TestDeepgramSTT:
    def test_transcribe_empty_returns_empty(self):
        stt = DeepgramSTT(api_key="test-key")
        assert stt.transcribe(b"") == ""

    def test_error_without_api_key(self):
        stt = DeepgramSTT(api_key="")
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RuntimeError, match="Deepgram API key"):
                stt.transcribe(b"audio")


# =============================================================================
# VAD tests
# =============================================================================


class TestSileroVAD:
    def test_init_defaults(self):
        vad = SileroVAD()
        assert vad.threshold == 0.5
        assert vad.min_speech_ms == 250
        assert vad.min_silence_ms == 300

    def test_custom_params(self):
        vad = SileroVAD(threshold=0.7, min_speech_ms=100, min_silence_ms=500)
        assert vad.threshold == 0.7
        assert vad.min_speech_ms == 100
        assert vad.min_silence_ms == 500

    def test_detect_speech_empty_audio(self):
        vad = SileroVAD()
        result = vad.detect_speech(b"")
        assert result == []

    def test_is_speech_empty_returns_false(self):
        vad = SileroVAD()
        assert vad.is_speech(b"") is False

    def test_energy_fallback_detects_loud_audio(self):
        """Energy-based fallback should detect loud audio as speech."""
        vad = SileroVAD()
        vad._available = False  # Force energy fallback
        loud_audio = _make_pcm_audio(duration_ms=100)
        assert vad.is_speech(loud_audio) is True

    def test_energy_fallback_detects_silence(self):
        """Energy-based fallback should classify silence as non-speech."""
        vad = SileroVAD()
        vad._available = False
        silent_audio = _make_silent_audio(duration_ms=100)
        assert vad.is_speech(silent_audio) is False

    def test_detect_speech_returns_segments(self):
        """detect_speech should return speech segments from mixed audio."""
        vad = SileroVAD(min_speech_ms=30, min_silence_ms=30)
        vad._available = False  # Use energy fallback

        # Build audio: speech + silence + speech
        speech1 = _make_pcm_audio(duration_ms=200)
        silence = _make_silent_audio(duration_ms=200)
        speech2 = _make_pcm_audio(duration_ms=200)
        audio = speech1 + silence + speech2

        segments = vad.detect_speech(audio)
        # Should find at least one speech segment
        assert len(segments) >= 1
        # Total speech bytes should be less than total audio
        total_speech = sum(len(s) for s in segments)
        assert total_speech <= len(audio)


# =============================================================================
# TTS provider tests
# =============================================================================


class TestVoxtralTTS:
    def test_synthesize_returns_wav_bytes(self):
        tts = VoxtralTTS()
        result = tts.synthesize("Hello world")
        # Should return bytes (WAV header starts with RIFF)
        assert isinstance(result, bytes)
        assert len(result) > 0
        assert result[:4] == b"RIFF"

    def test_synthesize_empty_returns_silence(self):
        tts = VoxtralTTS()
        result = tts.synthesize("")
        assert isinstance(result, bytes)
        assert result[:4] == b"RIFF"

    def test_stream_returns_chunks(self):
        tts = VoxtralTTS()
        chunks = []
        async def collect():
            async for chunk in tts.stream("Hello world"):
                chunks.append(chunk)
        asyncio.run(collect())
        assert len(chunks) >= 1


class TestKokoroTTS:
    def test_graceful_error_when_not_installed(self):
        tts = KokoroTTS()
        with patch.dict("sys.modules", {"kokoro": None}):
            with pytest.raises(RuntimeError, match="kokoro is not installed"):
                tts.synthesize("Hello")

    def test_model_lazy_loaded(self):
        tts = KokoroTTS()
        assert tts._model is None


class TestElevenLabsTTS:
    def test_error_without_api_key(self):
        tts = ElevenLabsTTS(api_key="")
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RuntimeError, match="ElevenLabs API key"):
                tts.synthesize("Hello")


class TestOpenAITTS:
    def test_error_without_api_key(self):
        tts = OpenAITTS(api_key="")
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RuntimeError, match="OpenAI API key"):
                tts.synthesize("Hello")


class TestPiperTTS:
    def test_graceful_error_when_not_installed(self):
        tts = PiperTTS()
        with patch.dict("sys.modules", {"piper": None, "piper.download": None}):
            with pytest.raises(RuntimeError, match="piper-tts is not installed"):
                tts.synthesize("Hello")

    def test_model_lazy_loaded(self):
        """Model should not be loaded at construction time."""
        tts = PiperTTS()
        assert tts._voice is None

    def test_custom_model_name(self):
        tts = PiperTTS(model="en_US-amy-low")
        assert tts._model_name == "en_US-amy-low"

    def test_synthesize_empty_returns_silence(self):
        tts = PiperTTS()
        result = tts.synthesize("")
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_synthesize_whitespace_returns_silence(self):
        tts = PiperTTS()
        result = tts.synthesize("   ")
        assert isinstance(result, bytes)
        assert len(result) > 0


class TestEdgeTTS:
    def test_graceful_error_when_not_installed(self):
        tts = EdgeTTS()
        with patch.dict("sys.modules", {"edge_tts": None}):
            with pytest.raises(RuntimeError, match="edge-tts is not installed"):
                tts.synthesize("Hello")

    def test_custom_voice(self):
        tts = EdgeTTS(voice="en-US-GuyNeural")
        assert tts._voice == "en-US-GuyNeural"

    def test_default_voice(self):
        tts = EdgeTTS()
        assert tts._voice == "en-US-AriaNeural"

    def test_synthesize_empty_returns_silence(self):
        tts = EdgeTTS()
        result = tts.synthesize("")
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_synthesize_whitespace_returns_silence(self):
        tts = EdgeTTS()
        result = tts.synthesize("   ")
        assert isinstance(result, bytes)
        assert len(result) > 0


class TestDetectBestTTS:
    def test_detects_kokoro_first(self):
        """Should prefer kokoro when available."""
        mock_kokoro = MagicMock()
        with patch.dict("sys.modules", {"kokoro": mock_kokoro}):
            provider = detect_best_tts()
            assert isinstance(provider, KokoroTTS)

    def test_falls_back_to_piper(self):
        """Should try piper when kokoro is unavailable."""
        mock_piper = MagicMock()
        with patch.dict(
            "sys.modules",
            {"kokoro": None, "piper": mock_piper},
        ):
            provider = detect_best_tts()
            assert isinstance(provider, PiperTTS)

    def test_falls_back_to_edge_tts(self):
        """Should try edge-tts when kokoro and piper are unavailable."""
        mock_edge = MagicMock()
        with patch.dict(
            "sys.modules",
            {"kokoro": None, "piper": None, "edge_tts": mock_edge},
        ):
            provider = detect_best_tts()
            assert isinstance(provider, EdgeTTS)

    def test_raises_when_nothing_available(self):
        """Should raise RuntimeError when no TTS library is installed."""
        with patch.dict(
            "sys.modules",
            {"kokoro": None, "piper": None, "edge_tts": None},
        ):
            with pytest.raises(RuntimeError, match="No TTS provider available"):
                detect_best_tts()


class TestGenerateSilence:
    def test_silence_length(self):
        silence = _generate_silence(duration_ms=100, sample_rate=16000)
        expected_samples = int(16000 * 100 / 1000)
        assert len(silence) == expected_samples * 2  # int16 = 2 bytes

    def test_silence_is_zeros(self):
        silence = _generate_silence(duration_ms=10)
        assert all(b == 0 for b in silence)


# =============================================================================
# VoicePipeline tests
# =============================================================================


class TestVoicePipeline:
    def test_init_with_string_providers(self):
        agent = MockAgent()
        pipeline = VoicePipeline(
            agent=agent,
            stt=MockSTT(),
            tts=MockTTS(),
        )
        assert isinstance(pipeline.stt, MockSTT)
        assert isinstance(pipeline.tts, MockTTS)

    def test_init_with_config(self):
        agent = MockAgent()
        config = VoiceConfig(
            activation="voice-activity",
            edge_mode=True,
            vad_threshold=0.7,
        )
        pipeline = VoicePipeline(
            agent=agent,
            stt=MockSTT(),
            tts=MockTTS(),
            config=config,
        )
        assert pipeline.activation == "voice-activity"
        assert pipeline._edge_mode is True
        assert pipeline.vad.threshold == 0.7

    def test_process_full_flow(self):
        """Pipeline should orchestrate STT -> Agent -> TTS."""
        agent = MockAgent(answer="The answer is 42.")
        stt = MockSTT(transcript="What is the meaning of life?")
        tts = MockTTS(audio=b"AUDIO_RESPONSE")

        pipeline = VoicePipeline(agent=agent, stt=stt, tts=tts, edge_mode=True)

        audio_input = _make_pcm_audio(duration_ms=500)
        result = asyncio.run(pipeline.process(audio_input))

        assert isinstance(result, VoiceResult)
        assert result.transcript == "What is the meaning of life?"
        assert result.answer == "The answer is 42."
        assert result.audio_out == b"AUDIO_RESPONSE"
        assert len(result.citations) >= 1

    def test_process_captures_latency(self):
        agent = MockAgent()
        pipeline = VoicePipeline(
            agent=agent, stt=MockSTT(), tts=MockTTS(), edge_mode=True
        )
        audio = _make_pcm_audio()
        result = asyncio.run(pipeline.process(audio))
        assert "vad_ms" in result.latency_ms
        assert "stt_ms" in result.latency_ms
        assert "llm_ms" in result.latency_ms
        assert "tts_ms" in result.latency_ms
        assert "total_ms" in result.latency_ms
        # All latencies should be non-negative
        for key, val in result.latency_ms.items():
            assert val >= 0, f"{key} should be >= 0"

    def test_process_empty_transcript(self):
        """Pipeline should return empty result if STT produces nothing."""
        agent = MockAgent()
        stt = MockSTT(transcript="")
        tts = MockTTS()

        pipeline = VoicePipeline(agent=agent, stt=stt, tts=tts, edge_mode=True)
        result = asyncio.run(pipeline.process(_make_pcm_audio()))

        assert result.transcript == ""
        assert result.answer == ""
        assert result.audio_out == b""

    def test_process_with_vad(self):
        """Pipeline with VAD enabled should still produce results."""
        agent = MockAgent()
        stt = MockSTT(transcript="test with vad")
        tts = MockTTS()

        pipeline = VoicePipeline(
            agent=agent, stt=stt, tts=tts, edge_mode=False
        )
        # Force energy-based VAD fallback
        pipeline._vad._available = False

        audio = _make_pcm_audio(duration_ms=500)
        result = asyncio.run(pipeline.process(audio))

        assert result.transcript == "test with vad"
        assert "vad_ms" in result.latency_ms

    def test_edge_mode_skips_vad(self):
        """Edge mode should skip VAD processing."""
        agent = MockAgent()
        stt = MockSTT()
        tts = MockTTS()

        pipeline = VoicePipeline(
            agent=agent, stt=stt, tts=tts, edge_mode=True
        )

        # Mock the VAD to track if it was called
        vad_mock = MagicMock()
        pipeline._vad = vad_mock

        audio = _make_pcm_audio()
        asyncio.run(pipeline.process(audio))

        # VAD's detect_speech should NOT be called in edge mode
        vad_mock.detect_speech.assert_not_called()

    def test_stream_produces_chunks(self):
        """stream() should yield audio chunks incrementally."""
        agent = MockAgent(answer="Hello. World.")
        stt = MockSTT(transcript="test")
        tts = MockTTS(audio=b"CHUNK_DATA_12345678")

        pipeline = VoicePipeline(
            agent=agent, stt=stt, tts=tts, edge_mode=True
        )

        chunks = []
        async def collect():
            async for chunk in pipeline.stream(_make_pcm_audio()):
                chunks.append(chunk)

        asyncio.run(collect())
        assert len(chunks) >= 1
        # All chunks should be bytes
        for chunk in chunks:
            assert isinstance(chunk, bytes)

    def test_stream_empty_transcript_yields_nothing(self):
        """stream() should yield nothing if STT produces empty text."""
        agent = MockAgent()
        stt = MockSTT(transcript="")
        tts = MockTTS()

        pipeline = VoicePipeline(
            agent=agent, stt=stt, tts=tts, edge_mode=True
        )

        chunks = []
        async def collect():
            async for chunk in pipeline.stream(_make_pcm_audio()):
                chunks.append(chunk)

        asyncio.run(collect())
        assert len(chunks) == 0

    def test_activation_modes(self):
        """Pipeline should accept all valid activation modes."""
        agent = MockAgent()
        for mode in ("push-to-talk", "voice-activity", "wake-word"):
            pipeline = VoicePipeline(
                agent=agent, stt=MockSTT(), tts=MockTTS(),
                activation=mode,
            )
            assert pipeline.activation == mode

    def test_init_stt_web_speech(self):
        """Pipeline should resolve 'web-speech' to WebSpeechSTT."""
        agent = MockAgent()
        pipeline = VoicePipeline(agent=agent, stt="web-speech", tts=MockTTS())
        assert isinstance(pipeline.stt, WebSpeechSTT)

    def test_init_tts_voxtral(self):
        """Pipeline should resolve 'voxtral' to VoxtralTTS."""
        agent = MockAgent()
        pipeline = VoicePipeline(agent=agent, stt=MockSTT(), tts="voxtral")
        assert isinstance(pipeline.tts, VoxtralTTS)

    def test_init_unknown_stt_falls_back(self):
        """Unknown STT name should fall back to FasterWhisperSTT."""
        agent = MockAgent()
        pipeline = VoicePipeline(agent=agent, stt="nonexistent", tts=MockTTS())
        assert isinstance(pipeline.stt, FasterWhisperSTT)

    def test_init_tts_piper(self):
        """Pipeline should resolve 'piper' to PiperTTS."""
        agent = MockAgent()
        pipeline = VoicePipeline(agent=agent, stt=MockSTT(), tts="piper")
        assert isinstance(pipeline.tts, PiperTTS)

    def test_init_tts_edge(self):
        """Pipeline should resolve 'edge-tts' to EdgeTTS."""
        agent = MockAgent()
        pipeline = VoicePipeline(agent=agent, stt=MockSTT(), tts="edge-tts")
        assert isinstance(pipeline.tts, EdgeTTS)

    def test_init_tts_auto(self):
        """Pipeline should resolve 'auto' via detect_best_tts."""
        agent = MockAgent()
        mock_kokoro = MagicMock()
        with patch.dict("sys.modules", {"kokoro": mock_kokoro}):
            pipeline = VoicePipeline(agent=agent, stt=MockSTT(), tts="auto")
            assert isinstance(pipeline.tts, KokoroTTS)

    def test_init_unknown_tts_falls_back(self):
        """Unknown TTS name should fall back via auto-detect."""
        agent = MockAgent()
        # With kokoro importable, auto-detect returns KokoroTTS
        mock_kokoro = MagicMock()
        with patch.dict("sys.modules", {"kokoro": mock_kokoro}):
            pipeline = VoicePipeline(agent=agent, stt=MockSTT(), tts="nonexistent")
            assert isinstance(pipeline.tts, (KokoroTTS, PiperTTS, EdgeTTS))


# =============================================================================
# Sentence splitting tests
# =============================================================================


class TestSentenceSplitting:
    def test_split_simple(self):
        sentences = _split_sentences("Hello world. How are you? Fine!")
        assert sentences == ["Hello world.", "How are you?", "Fine!"]

    def test_split_single_sentence(self):
        sentences = _split_sentences("Just one sentence.")
        assert sentences == ["Just one sentence."]

    def test_split_empty(self):
        sentences = _split_sentences("")
        assert sentences == []

    def test_split_no_punctuation(self):
        sentences = _split_sentences("no punctuation here")
        assert sentences == ["no punctuation here"]


# =============================================================================
# VoiceResult dataclass tests
# =============================================================================


class TestVoiceResult:
    def test_defaults(self):
        result = VoiceResult(transcript="hi", answer="hello", audio_out=b"audio")
        assert result.transcript == "hi"
        assert result.answer == "hello"
        assert result.audio_out == b"audio"
        assert result.citations == []
        assert result.latency_ms == {}

    def test_with_latency(self):
        latency = {"stt_ms": 50.0, "llm_ms": 100.0, "tts_ms": 30.0, "total_ms": 180.0}
        result = VoiceResult(
            transcript="test",
            answer="response",
            audio_out=b"data",
            latency_ms=latency,
        )
        assert result.latency_ms["total_ms"] == 180.0


# =============================================================================
# Integration: SaidoAgent voice_config parameter
# =============================================================================


class TestSaidoAgentVoiceConfig:
    """Test that SaidoAgent accepts voice_config and exposes voice_pipeline."""

    def test_agent_accepts_voice_config(self):
        """SaidoAgent.__init__ should accept voice_config parameter."""
        from saido_agent import SaidoAgent

        # We just check that the parameter is accepted without error.
        # We can't fully init SaidoAgent without a knowledge dir, but we
        # can check the signature accepts the param.
        import inspect
        sig = inspect.signature(SaidoAgent.__init__)
        assert "voice_config" in sig.parameters


# =============================================================================
# __init__.py exports
# =============================================================================


class TestVoiceExports:
    def test_pipeline_importable(self):
        from saido_agent.voice import VoicePipeline
        assert VoicePipeline is not None

    def test_config_importable(self):
        from saido_agent.voice import VoiceConfig
        assert VoiceConfig is not None

    def test_result_importable(self):
        from saido_agent.voice import VoiceResult
        assert VoiceResult is not None

    def test_stt_providers_importable(self):
        from saido_agent.voice import STTProvider, FasterWhisperSTT, DeepgramSTT, WebSpeechSTT
        assert STTProvider is not None
        assert FasterWhisperSTT is not None

    def test_tts_providers_importable(self):
        from saido_agent.voice import (
            TTSProvider, KokoroTTS, PiperTTS, EdgeTTS,
            VoxtralTTS, ElevenLabsTTS, OpenAITTS,
            detect_best_tts,
        )
        assert TTSProvider is not None
        assert PiperTTS is not None
        assert EdgeTTS is not None
        assert detect_best_tts is not None

    def test_vad_importable(self):
        from saido_agent.voice import SileroVAD
        assert SileroVAD is not None

    def test_legacy_exports(self):
        from saido_agent.voice import (
            check_voice_deps,
            voice_input,
            transcribe,
            check_stt_availability,
        )
        assert check_voice_deps is not None
        assert voice_input is not None
