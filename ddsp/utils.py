import numpy as np
import matplotlib.pyplot as plt
import librosa as li

def get_scheduler(len_dataset, start_lr, stop_lr, length):
    def schedule(epoch):
        step = epoch * len_dataset
        if step < length:
            t = step / length
            return start_lr * (1 - t) + stop_lr * t
        else:
            return stop_lr

    return schedule


def plot_sig(sig, stem=False):
    """ plot a 1d signal """
    plt.close('all')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))

    if stem:
        ax.stem(sig)
    else:
        ax.plot(sig)

    ax.set_xlabel('sample')

    return fig, ax


def plot_spec(stft):
    """ returns a fig and an ax"""
    plt.close('all')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))
    pr = lambda m: li.amplitude_to_db(m)
    ax.imshow(pr(stft), aspect='auto')
    ax.invert_yaxis()

    return fig, ax

def hz_to_midi(freqs):
    return 12 * np.log2(freqs / 440) + 69

def midi_to_hz(midi):
    return 440 * 2  ** ((midi - 69) / 12)

def tonp(tensor):
    return tensor.detach().cpu().numpy()

IDX = 0

def log_sample_stft(writer, stft, tag, step, config, mel=True):
    # use a medium scale for multi-scale stft
    scl_idx = len(stft[IDX])//2
    stft = tonp(stft[IDX][scl_idx])

    if mel:
        sr = config['preprocess']["sampling_rate"]
        n_fft = config["train"]["scales"][IDX]
        hop_length = config["train"]["overlap"]
        stft = li.feature.melspectrogram(S=stft, sr=sr, n_fft=n_fft, hop_length=hop_length)

    fig, ax = plot_spec(stft)
    writer.add_figure(tag, fig, step)
    return fig

def log_harmonic_amps(writer, amps, tag, step):
    amps = tonp(amps[IDX].T)
    fig, ax = plot_spec(amps)
    writer.add_figure(tag, fig, step)
    return fig

def log_pitch_curve(writer, pitches, tag, step):
    freqs = tonp(pitches[IDX].squeeze(-1))
    midi = hz_to_midi(freqs)
    fig, ax = plot_sig(midi)
    ax.set_ylim([-1, 128])
    writer.add_figure(tag, fig, step)

def log_loudness_curve(writer, loudness, tag, step):
    loud = tonp(loudness[IDX].squeeze(-1))
    fig, ax = plot_sig(loud)
    writer.add_figure(tag, fig, step)