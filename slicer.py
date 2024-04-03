import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


def auto_slice(sample_path):
    full_sample, sample_rate = librosa.load(sample_path)

    # Onset Params
    pre_max = 5
    post_max = 30
    wait = 30

    onset_times = librosa.onset.onset_detect(y=full_sample, sr=sample_rate, backtrack=True, units='time',
                                             wait=wait, pre_max=pre_max, post_max=post_max)

    # Graph to check slice locations
    x = 10
    short_full_sample = full_sample[::x]
    fig, ax = plt.subplots()
    x_times = np.linspace(0, full_sample.size / sample_rate, num=short_full_sample.size, endpoint=False)
    ax.plot(x_times, short_full_sample, label='Signal')
    ax.vlines(onset_times, 0, short_full_sample.max(), colors='r', linestyles='dashed', label='Slice Points')
    ax.legend()
    plt.show()

    # Output
    # onset_times = np.append(onset_times, (full_sample.size - 1) / sample_rate)
    # for i in range(6):
    #     for j in range(16):
    #         onset_frames = onset_times * sample_rate
    #         small_sample = full_sample[onset_frames[16 * i + j]:onset_frames[16 * i + j + 1]]
    #         sf.write(f'../alliestringsamples/s{i + 1}n{j}.wav',
    #                  small_sample, int(sample_rate))


samp_path = '../alliestringsamples/to15-allstrings-allie-acetal-neck.wav'
auto_slice(samp_path)
