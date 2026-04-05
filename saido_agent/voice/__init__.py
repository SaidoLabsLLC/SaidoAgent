"""Voice package for Saido Agent.

Public API surface:
  - ``VoicePipeline``  -- bidirectional voice pipeline (STT -> Agent -> TTS)
  - ``VoiceConfig``    -- configuration dataclass
  - ``VoiceResult``    -- pipeline output dataclass
  - ``STTProvider``    -- STT provider ABC
  - ``TTSProvider``    -- TTS provider ABC

Legacy API (from nano-claude-code port):
  - ``voice_input()``  -- record-and-transcribe convenience function
  - ``check_voice_deps()`` -- dependency availability check
"""

from saido_agent.voice.config import VoiceConfig
from saido_agent.voice.pipeline import VoicePipeline, VoiceResult
from saido_agent.voice.stt import (
    STTProvider,
    DeepgramSTT,
    FasterWhisperSTT,
    WebSpeechSTT,
    check_stt_availability,
    transcribe,
)
from saido_agent.voice.tts import (
    TTSProvider,
    ElevenLabsTTS,
    KokoroTTS,
    OpenAITTS,
    VoxtralTTS,
)
from saido_agent.voice.vad import SileroVAD
from saido_agent.voice.recorder import check_recording_availability, record_until_silence
from saido_agent.voice.keyterms import get_voice_keyterms


def check_voice_deps() -> tuple[bool, str | None]:
    """Return (available, reason_if_not)."""
    rec_ok, rec_reason = check_recording_availability()
    if not rec_ok:
        return False, rec_reason
    stt_ok, stt_reason = check_stt_availability()
    if not stt_ok:
        return False, stt_reason
    return True, None


def voice_input(
    language: str = "auto",
    max_seconds: int = 30,
    on_energy: "callable | None" = None,
) -> str:
    """Record until silence, then transcribe."""
    keyterms = get_voice_keyterms()
    pcm = record_until_silence(max_seconds=max_seconds, on_energy=on_energy)
    if not pcm:
        return ""
    return transcribe(pcm, keyterms=keyterms, language=language)


__all__ = [
    # New voice pipeline API
    "VoicePipeline",
    "VoiceConfig",
    "VoiceResult",
    "STTProvider",
    "TTSProvider",
    "FasterWhisperSTT",
    "DeepgramSTT",
    "WebSpeechSTT",
    "KokoroTTS",
    "VoxtralTTS",
    "ElevenLabsTTS",
    "OpenAITTS",
    "SileroVAD",
    # Legacy API
    "check_voice_deps",
    "check_recording_availability",
    "check_stt_availability",
    "record_until_silence",
    "transcribe",
    "get_voice_keyterms",
    "voice_input",
]
