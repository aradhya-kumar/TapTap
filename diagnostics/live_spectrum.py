import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from audio.stream import AudioStream


latest = np.zeros(512)


def callback(indata, frames, time_info, status):
    global latest

    latest = indata[:, 0].copy()


stream = AudioStream()

stream.start(callback)

fig, ax = plt.subplots(figsize=(12,4))

freqs = np.fft.rfftfreq(512, 1/48000)

line, = ax.plot(freqs, np.zeros(len(freqs)))

ax.set_xlim(0,10000)
ax.set_ylim(0,5)

ax.set_title("Live Spectrum")


def update(frame):

    fft = np.abs(np.fft.rfft(latest))

    line.set_ydata(fft)

    return line,


ani = FuncAnimation(
    fig,
    update,
    interval=20,
    blit=True
)

plt.show()

stream.stop()