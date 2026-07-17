import numpy as np

from dsp.features import extract

signal = np.random.randn(4800)

features = extract(signal, 48000)

for key, value in features.items():

    print(key)

    print(value)

    print()