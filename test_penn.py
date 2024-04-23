import audioplot
import penn
import torch

# CONFIG
# Here we'll use a 10 millisecond hopsize
hopsize = .01

# Provide a sensible frequency range given your domain and model
fmin = 30.
fmax = 1500.

# Choose a gpu index to use for inference. Set to None to use cpu.
gpu = None

# If you are using a gpu, pick a batch size that doesn't cause memory errors
# on your gpu
batch_size = 2048

# Select a checkpoint to use for inference. The default checkpoint will
# download and use FCNF0++ pretrained on MDB-stem-synth and PTDB
checkpoint = None

# Centers frames at hopsize / 2, 3 * hopsize / 2, 5 * hopsize / 2, ...
center = 'half-hop'

# (Optional) Linearly interpolate unvoiced regions below periodicity threshold
interp_unvoiced_at = .065

# SOURCE AUDIO
# audio = penn.load.audio('../../Data/test/assets/gershwin.wav')
# audio_path = '../Datasets/GuitarSet/audio_hex-pickup_original/02_Rock1-90-C#_solo_hex.wav'
audio_path = '../Datasets/GuitarSet/audio_mono-mic/02_Rock1-90-C#_solo_mic.wav'
# Load audio at the correct sample rate
audio = penn.load.audio(audio_path)
num_channels, num_frames = audio.shape

# MONOPHONIC
if num_channels == 1:
    # reshape to torch.Size([1, x])
    # audio = audio.unsqueeze(0)
    pitch, periodicity = penn.from_audio(
        audio,
        penn.SAMPLE_RATE,
        hopsize=hopsize,
        fmin=fmin,
        fmax=fmax,
        checkpoint=checkpoint,
        batch_size=batch_size,
        center=center,
        interp_unvoiced_at=interp_unvoiced_at,
        gpu=gpu)

    audioplot.plot_pitch_over_specgram(
            pitch, periodicity, hopsize, [fmin, fmax], audio, penn.SAMPLE_RATE)

# HEXAPHONIC
if num_channels == 6:
    pitch6 = []
    period6 = []

    for channel in range(num_channels):
        pitch, periodicity = penn.from_audio(
            audio[channel].unsqueeze(0),
            penn.SAMPLE_RATE,
            hopsize=hopsize,
            fmin=fmin,
            fmax=fmax,
            checkpoint=checkpoint,
            batch_size=batch_size,
            center=center,
            interp_unvoiced_at=interp_unvoiced_at,
            gpu=gpu)

        pitch6.append(pitch)
        period6.append(periodicity)

    pitch6 = torch.stack(pitch6)
    period6 = torch.stack(period6)

    # turn shape [6, 1, x] into [6, x]
    pitch6 = torch.squeeze(pitch6)
    period6 = torch.squeeze(period6)

    print(f"pitch6 shape: {pitch6.shape}")
    print(f"period6 shape: {period6.shape}")

    audioplot.plot_pitch_over_specgram6(pitch6, period6, hopsize, [fmin, fmax],
                                        audio, penn.SAMPLE_RATE)

input("Press any key to continue: ")
