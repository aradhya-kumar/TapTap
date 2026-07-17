import numpy as np
import soundfile as sf

from audio.stream import AudioStream


BUFFER = []

WINDOW = 200


def callback(indata, frames, time_info, status):

    global BUFFER

    BUFFER.extend(indata[:,0])

    if len(BUFFER) > 48000:
        BUFFER = BUFFER[-48000:]

    peak = np.max(np.abs(indata))

    if peak > 0.08:

        print("Impulse detected!")

        audio = np.array(BUFFER[-WINDOW*5:])

        sf.write("debug/impulse.wav", audio, 48000)

        print("Saved debug/impulse.wav")


stream = AudioStream()

stream.start(callback)

print("Listening...")

input()

stream.stop()