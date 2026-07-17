import time
from collections import deque
from pathlib import Path

import numpy as np
import soundfile as sf

from audio.stream import AudioStream


class TapDetector:

    def __init__(
        self,
        samplerate=48000,
        pre_ms=45,
        post_ms=45,
        cooldown_ms=200,
        warmup_sec=1.0,
    ):

        self.fs = samplerate

        self.pre_samples = int(pre_ms * samplerate / 1000)
        self.post_samples = int(post_ms * samplerate / 1000)

        self.cooldown = cooldown_ms / 1000

        self.last_detection = 0

        self.buffer = deque(maxlen=self.pre_samples * 4)

        self.collecting = False
        self.remaining = 0
        self.event = []

        self.noise_samples = []
        self.warmup = warmup_sec

        self.threshold = None

        self.save_folder = Path("data/raw")
        self.save_folder.mkdir(parents=True, exist_ok=True)

        self.counter = 0

    def process(self, audio):

        samples = audio[:, 0]

        for sample in samples:

            self.buffer.append(sample)

        rms = np.sqrt(np.mean(samples ** 2))
        peak = np.max(np.abs(samples))

        # ------------------------
        # Learn room noise
        # ------------------------

        if self.threshold is None:

            self.noise_samples.append(rms)

            if len(self.noise_samples) * len(samples) >= self.fs * self.warmup:

                noise = np.mean(self.noise_samples)

                self.threshold = noise * 6

                print(f"\nNoise Floor : {noise:.6f}")
                print(f"Threshold   : {self.threshold:.6f}")
                print("\nListening...\n")

            return

        # ------------------------
        # Already recording?
        # ------------------------

        if self.collecting:

            self.event.extend(samples)

            self.remaining -= len(samples)

            if self.remaining <= 0:

                self.save_event()

                self.collecting = False

            return

        # ------------------------
        # Cooldown
        # ------------------------

        if time.time() - self.last_detection < self.cooldown:
            return

        # ------------------------
        # Tap Detection
        # ------------------------

        if peak > self.threshold:

            print(f"Tap detected! Peak={peak:.3f}")

            self.last_detection = time.time()

            self.collecting = True

            self.remaining = self.post_samples

            self.event = list(self.buffer)

    def save_event(self):

        audio = np.array(self.event)

        filename = self.save_folder / f"tap_{self.counter:04}.wav"

        sf.write(filename, audio, self.fs)

        print(f"Saved {filename}")

        self.counter += 1


detector = TapDetector()


stream = AudioStream()

stream.start(detector.process)

print("Learning room noise...")

try:

    while True:
        time.sleep(1)

except KeyboardInterrupt:

    stream.stop()