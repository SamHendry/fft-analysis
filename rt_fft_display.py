import numpy as np
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pyaudio import PyAudio, paInt16

def main():
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

    fig, ax = plt.subplots(num='Frequency Spectrum')
    line, = ax.plot(np.zeros(chunk))
    freqbins = rfftfreq(chunk, 1/sample_rate)
    hamming_window = np.hamming(chunk)


    def init():
        ax.set_ylim(0, 5000)
        ax.set_xlim(0, 1600)
        ax.set_ylabel('Magnitude')
        ax.set_xlabel('Hz')
        return line,


    def update_data(_):
        data = np.frombuffer(stream.read(chunk), dtype=np.int16)
        data_ham = hamming_window * data
        data_mags_norm = np.abs(rfft(data_ham)) / chunk
        line.set_data(freqbins, data_mags_norm)
        return line,


    _ = FuncAnimation(fig, update_data, interval=25, init_func=init, blit=True, cache_frame_data=False)
    plt.show()

if __name__ == "__main__":
    main()