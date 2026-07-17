import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from audio.stream import AudioStream


latest = np.zeros(512)


def callback(indata, frames, time_info, status):
    global latest

    if status:
        print(status)

    latest = indata[:, 0].copy()


stream = AudioStream()

stream.start(callback)

fig, ax = plt.subplots(figsize=(12, 4))

line, = ax.plot(latest)

ax.set_ylim(-0.1, 0.1)
ax.set_xlim(0, 512)

ax.set_title("EchoDesk Live Waveform")


def update(frame):

    line.set_ydata(latest)

    return line,


ani = FuncAnimation(
    fig,
    update,
    interval=10,
    blit=True
)

plt.show()

stream.stop()