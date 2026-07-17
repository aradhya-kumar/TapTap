import sounddevice as sd


class AudioStream:

    def __init__(
        self,
        samplerate=48000,
        channels=1,
        blocksize=512,
        device=1
    ):

        self.samplerate = samplerate
        self.channels = channels
        self.blocksize = blocksize
        self.device = device

        self.stream = None
        self.callback_function = None

    def _callback(self, indata, frames, time_info, status):

        if status:
            print(status)

        if self.callback_function is not None:
            self.callback_function(indata.copy())

    def start(self, callback):

        self.callback_function = callback

        self.stream = sd.InputStream(
            device=self.device,
            samplerate=self.samplerate,
            channels=self.channels,
            blocksize=self.blocksize,
            dtype="float32",
            callback=self._callback,
        )

        self.stream.start()

    def stop(self):

        if self.stream is not None:
            self.stream.stop()
            self.stream.close()