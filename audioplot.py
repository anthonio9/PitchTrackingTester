import torch
import matplotlib.pyplot as plt
import warnings

plt.ion()


def plot_waveform(waveform, sample_rate):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("waveform")
    plt.show(block=False)


def plot_specgram(waveform, sample_rate, title="Spectrogram"):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle(title)
    plt.show(block=False)


def plot_pitch_over_specgram6(
        pitch, periodicity, hopsize, pitch_range, waveform, sample_rate):

    pitch = pitch.numpy()
    waveform = waveform.numpy()

    num_channels, num_frames = pitch.shape
    _, wave_num_frames = waveform.shape

    if num_channels != 6:
        warnings.warn("Number of provided channels isn't equal to 6!")
        return

    pitch_time_axis = torch.arange(0, num_frames)*hopsize
    waveform_time_axis = torch.arange(0, wave_num_frames) / sample_rate

    # extend the upper limit of the plotted pitch just slightly
    pitch_range[1] = pitch.max() * 5/4

    figure, axes = plt.subplots(num_channels, 2)

    for j in range(2):
        for i in range(0, num_channels, 2):
            # calculate the index
            ind = j*3 + i // 2

            print(f"waveform[{ind}].shape: {waveform[ind].shape}")

            # plot the waveform
            axes[i, j].plot(waveform_time_axis, waveform[ind], color='b')
            axes[i, j].set_xlabel("Time [s]")
            axes[i, j].set_title(f"Waveform: Periodicity [{ind}]")

            # plot the periodicity on the waveform
            color_twin = 'tab:orange'
            axes_twin = axes[i, j].twinx()
            axes_twin.plot(pitch_time_axis, periodicity[ind], color=color_twin,
                           linewidth=2, linestyle='dashed')
            axes_twin.set_ylabel('Periodicity', color=color_twin)
            axes_twin.tick_params(axis='y', labelcolor=color_twin)

            # plot the Spectrogram and the pitch + periodicity
            color = 'tab:red'
            axes[i + 1, j].specgram(waveform[ind], Fs=sample_rate)
            axes[i + 1, j].plot(pitch_time_axis, pitch[ind], color=color,
                                linewidth=2)
            # axes[i + 1, j].scatter(pitch_time_axis, pitch[ind],
            #                        c=color, linewidth=2, marker='.')
            axes[i + 1, j].set_ylim(pitch_range)
            axes[i + 1, j].set_xlabel("Time [s]")
            axes[i + 1, j].set_ylabel("Frequency [Hz]", color=color)
            axes[i + 1, j].tick_params(axis='y', labelcolor=color)
            axes[i + 1, j].set_title(
                    f"Spectrogram: Pitch + Periodicity [{ind}]")

            # plot periodicity on the same x-axis with separate y-axis
            color_twin = 'tab:orange'
            axes_twin = axes[i + 1, j].twinx()
            axes_twin.plot(pitch_time_axis, periodicity[ind], color=color_twin,
                           linewidth=2, linestyle='dashed')
            axes_twin.set_ylabel('Periodicity', color=color_twin)
            axes_twin.tick_params(axis='y', labelcolor=color_twin)

    plt.show(block=False)


def plot_pitch_over_specgram(
        pitch, periodicity, hopsize, pitch_range, waveform, sample_rate):

    pitch = pitch.numpy()
    waveform = waveform.numpy()

    num_channels, num_frames = pitch.shape
    _, wave_num_frames = waveform.shape

    pitch_time_axis = torch.arange(0, num_frames)*hopsize
    waveform_time_axis = torch.arange(0, wave_num_frames) / sample_rate

    # extend the upper limit of the plotted pitch just slightly
    pitch_range[1] = pitch.max() * 5/4

    figure, axes = plt.subplots(num_channels*2, 1)

    # if num_channels == 1:
    #     axes = [axes]

    # plot the waveform
    axes[0].plot(waveform_time_axis, waveform[0], color='b')
    axes[0].set_xlabel("Time [s]")

    # plot the periodicity on the waveform
    color_twin = 'tab:orange'
    axes_twin = axes[0].twinx()
    axes_twin.plot(pitch_time_axis, periodicity[0], color=color_twin,
                   linewidth=2, linestyle='dashed')
    axes_twin.set_ylabel('Periodicity', color=color_twin)
    axes_twin.tick_params(axis='y', labelcolor=color_twin)

    # plot the Spectrogram
    color = 'tab:red'
    axes[1].specgram(waveform[0], Fs=sample_rate)
    # axes[1].plot(pitch_time_axis, pitch[0], color=color, linewidth=2,
    #              linestyle='dotted', marker='.')
    axes[1].scatter(pitch_time_axis, pitch[0], c=color, linewidth=2,
                    marker='.')
    axes[1].set_ylabel("Frequency [Hz]", color=color)
    axes[1].tick_params(axis='y', labelcolor=color)
    axes[1].set_ylim(pitch_range)

    # plot periodicity on the same x-axis with separate y-axis
    color_twin = 'tab:orange'
    axes_twin = axes[1].twinx()
    axes_twin.plot(pitch_time_axis, periodicity[0], color=color_twin,
                   linewidth=2, linestyle='dashed')
    axes_twin.set_ylabel('Periodicity', color=color_twin)
    axes_twin.tick_params(axis='y', labelcolor=color_twin)

    axes[1].set_xlabel("Time [s]")

    figure.suptitle("Spectrogram")
    plt.show(block=False)
