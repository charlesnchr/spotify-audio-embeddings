#!/usr/bin/env python3
"""
Create test audio files with different characteristics for embedding testing.
"""
import numpy as np
import soundfile as sf
from pathlib import Path
import librosa

# Create the previews directory
previews_dir = Path("previews")
previews_dir.mkdir(exist_ok=True)

def create_sine_wave(frequency, duration=30, sample_rate=22050):
    """Create a simple sine wave."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    wave = np.sin(2 * np.pi * frequency * t)
    return wave

def create_chord(frequencies, duration=30, sample_rate=22050):
    """Create a chord from multiple frequencies."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    wave = np.zeros_like(t)
    for freq in frequencies:
        wave += np.sin(2 * np.pi * freq * t)
    return wave / len(frequencies)

def create_noise(duration=30, sample_rate=22050, noise_type='white'):
    """Create different types of noise."""
    samples = int(sample_rate * duration)
    if noise_type == 'white':
        return np.random.normal(0, 0.1, samples)
    elif noise_type == 'pink':
        # Simple pink noise approximation
        white = np.random.normal(0, 0.1, samples)
        pink = librosa.util.normalize(np.cumsum(white))
        return pink
    else:
        return np.random.normal(0, 0.1, samples)

def add_envelope(wave, attack=0.1, decay=0.1, sustain=0.7, release=0.2):
    """Add ADSR envelope to a wave."""
    total_samples = len(wave)
    envelope = np.ones(total_samples)
    
    # Attack
    attack_samples = int(attack * total_samples)
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    
    # Decay
    decay_samples = int(decay * total_samples)
    start_idx = attack_samples
    end_idx = start_idx + decay_samples
    envelope[start_idx:end_idx] = np.linspace(1, sustain, decay_samples)
    
    # Sustain (already set to sustain level)
    sustain_samples = int((1 - attack - decay - release) * total_samples)
    start_idx = attack_samples + decay_samples
    end_idx = start_idx + sustain_samples
    envelope[start_idx:end_idx] = sustain
    
    # Release
    release_samples = int(release * total_samples)
    start_idx = total_samples - release_samples
    envelope[start_idx:] = np.linspace(sustain, 0, release_samples)
    
    return wave * envelope

# Create test audio files
sample_rate = 22050
duration = 30

print("Creating test audio files...")

# 1. Similar tracks - Different keys of the same chord progression
# C major chord
c_major = create_chord([261.63, 329.63, 392.00])  # C, E, G
c_major = add_envelope(c_major)
sf.write(previews_dir / "c_major_chord.mp3", c_major, sample_rate)

# G major chord (similar to C major)
g_major = create_chord([392.00, 493.88, 587.33])  # G, B, D
g_major = add_envelope(g_major)
sf.write(previews_dir / "g_major_chord.mp3", g_major, sample_rate)

# F major chord (similar to C major)
f_major = create_chord([349.23, 440.00, 523.25])  # F, A, C
f_major = add_envelope(f_major)
sf.write(previews_dir / "f_major_chord.mp3", f_major, sample_rate)

# 2. Different styles/genres
# Low frequency "bass" sound
bass_sound = create_sine_wave(80)  # Low bass note
bass_sound = add_envelope(bass_sound)
sf.write(previews_dir / "bass_sound.mp3", bass_sound, sample_rate)

# High frequency "bell" sound
bell_sound = create_sine_wave(1000)  # High bell note
bell_sound = add_envelope(bell_sound)
sf.write(previews_dir / "bell_sound.mp3", bell_sound, sample_rate)

# Complex harmonic sound (multiple overtones)
complex_sound = create_chord([220, 440, 880, 1320])  # A with overtones
complex_sound = add_envelope(complex_sound)
sf.write(previews_dir / "complex_harmonic.mp3", complex_sound, sample_rate)

# 3. Dissimilar tracks
# White noise
white_noise = create_noise(duration, sample_rate, 'white')
white_noise = add_envelope(white_noise)
sf.write(previews_dir / "white_noise.mp3", white_noise, sample_rate)

# Pink noise
pink_noise = create_noise(duration, sample_rate, 'pink')
pink_noise = add_envelope(pink_noise)
sf.write(previews_dir / "pink_noise.mp3", pink_noise, sample_rate)

print("Created test audio files:")
for file in sorted(previews_dir.glob("*.mp3")):
    print(f"  - {file.name}")