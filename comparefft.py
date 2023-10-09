import numpy as np
from scipy.fft import rfft
import librosa
import matplotlib.pyplot as plt


def get_mags(sample, normalize=True):
    # load sample -> normalize -> fft -> abs
    # n is kinda used to correct the freq bins
    # might not need to normalize if I deal only with peaks relatively

    sample, sr = librosa.load(sample, mono=True)
    if normalize:
        sample = librosa.util.normalize(sample)
    freq1 = rfft(sample, n=int(sr))
    return np.abs(freq1)


def freq_compare(low_samp, high_samp, cutoff=2400):
    mags1 = get_mags(low_samp)
    mags2 = get_mags(high_samp)

    # max
    print(np.argmax(mags1))

    # plotting
    fig, ax = plt.subplots()
    ax.plot(mags1[:cutoff], color="b", label="Low")
    ax.plot(mags2[:cutoff], color="r", label="High", linestyle=":")
    ax.set_xlabel("Hz")
    ax.set_ylabel("Normalized Magnitude")
    ax.set_title(f"Note Comparison (cutoff={cutoff})")
    ax.legend()
    plt.show()


def freq_ncompare(low_samp, high_samp, cutoff=2400):
    """Compare ffts of two samples, one with both samples normalized and one with both samples not normalized"""

    # low
    mags1 = get_mags(low_samp, normalize=False)
    mags1n = get_mags(low_samp, normalize=True)

    # high
    mags2 = get_mags(high_samp, normalize=False)
    mags2n = get_mags(high_samp, normalize=True)

    # plotting
    fig, axs = plt.subplots(2)
    axs[0].plot(mags1n[:cutoff], color="b", label="Low", linestyle=":")
    axs[0].plot(mags2n[:cutoff], color="r", label="High", linestyle=":")
    axs[0].set_xlabel("Hz")
    axs[0].set_ylabel("Magnitude")
    axs[0].set_title("Normalized")
    axs[0].legend()
    axs[1].plot(mags1[:cutoff], color="b", label="Low", linestyle=":")
    axs[1].plot(mags2[:cutoff], color="r", label="High", linestyle=":")
    axs[1].set_xlabel("Hz")
    axs[1].set_ylabel("Magnitude")
    axs[1].set_title("Standard")
    axs[1].legend()
    plt.show()


low_sample_path = "../alliestringsamples/s1n0.wav"
high_sample_path = "../alliestringsamples/s1n12.wav"
freq_compare(low_sample_path, high_sample_path, cutoff=1200)
