import numpy as np
from scipy.fft import rfft, rfftfreq
from pyaudio import PyAudio, paInt16

def peaks() -> None:
    sample_rate = 44100
    chunk = 2048
    p = PyAudio()

    stream = p.open(
        format=paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        output=True,
        frames_per_buffer=chunk
    )

    freqbins = rfftfreq(chunk, 1/sample_rate)
    hamming_window = np.hamming(chunk)

    while True: # keyboard interrupt will stop?
        # obtain data
        data = np.frombuffer(stream.read(chunk), dtype=np.int16)
        # process data
        data_ham = hamming_window * data
        data_mags_norm = np.abs(rfft(data_ham)) / chunk
        # TODO: find peaks

        # TODO: print peaks


if __name__ == "__main__":
    peaks()