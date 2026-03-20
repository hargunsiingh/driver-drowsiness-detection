"""
Alert Manager
Handles audio alarm playback for drowsiness alerts using pygame.
Generates a built-in alarm tone so no external sound files are needed.
"""

import io
import math
import struct
import wave

import pygame


def _generate_alarm_wav() -> bytes:
    """
    Generate a loud, pulsing alarm tone as a WAV byte buffer.
    Two-tone alternating beep pattern at 880Hz and 1100Hz.
    """
    sample_rate = 44100
    duration = 2.0  # seconds (will loop)
    volume = 0.8
    samples = []

    for i in range(int(sample_rate * duration)):
        t = i / sample_rate
        # Alternate between two tones every 0.25s for urgency
        freq = 880 if (t % 0.5) < 0.25 else 1100
        # Add a pulsing amplitude envelope
        envelope = 0.5 + 0.5 * math.sin(2 * math.pi * 4 * t)
        sample = volume * envelope * math.sin(2 * math.pi * freq * t)
        samples.append(int(sample * 32767))

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))

    buf.seek(0)
    return buf


class AlertManager:
    """Manages drowsiness alarm playback."""

    def __init__(self):
        pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
        wav_buffer = _generate_alarm_wav()
        self._alarm_sound = pygame.mixer.Sound(wav_buffer)
        self._alarm_sound.set_volume(0.9)
        self._playing = False

    def trigger_alarm(self):
        """Start playing the alarm (loops). Idempotent — safe to call repeatedly."""
        if not self._playing:
            self._alarm_sound.play(loops=-1)  # -1 = infinite loop
            self._playing = True

    def stop_alarm(self):
        """Stop the alarm. Idempotent."""
        if self._playing:
            self._alarm_sound.stop()
            self._playing = False

    @property
    def is_playing(self) -> bool:
        return self._playing

    def cleanup(self):
        """Release pygame resources."""
        self.stop_alarm()
        pygame.mixer.quit()
