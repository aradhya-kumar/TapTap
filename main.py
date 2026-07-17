# import time
# import numpy as np

# from audio.stream import AudioStream


# def callback(indata, frames, time_info, status):

#     if status:
#         print(status)

#     rms = np.sqrt(np.mean(indata**2))

#     bars = int(rms * 400)

#     print("█" * bars)


# stream = AudioStream()

# stream.start(callback)

# print("Listening...")

# try:
#     while True:
#         time.sleep(1)

# except KeyboardInterrupt:
#     stream.stop()

import sounddevice as sd

print(sd.default.device)