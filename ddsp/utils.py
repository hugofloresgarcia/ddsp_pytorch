import numpy as np
import matplotlib.pyplot as plt
import librosa as li
from torch.utils.tensorboard.writer import SummaryWriter

def get_scheduler(len_dataset, start_lr, stop_lr, length):
    def schedule(epoch):
        step = epoch * len_dataset
        if step < length:
            t = step / length
            return start_lr * (1 - t) + stop_lr * t
        else:
            return stop_lr

    return schedule


def plot_sig(sig, ax, stem=False):
    """ plot a 1d signal """

    if stem:
        ax.stem(sig)
    else:
        ax.plot(sig)

    ax.set_xlabel('sample')

    return ax

def plot_spec(stft, ax, amp_to_db=True):
    """ returns a fig and an ax"""
    pr = lambda m: li.amplitude_to_db(m)
    stft = pr(stft) if amp_to_db else stft
    im = ax.imshow(stft, aspect='auto')

    fig = plt.gcf()
    fig.colorbar(im, ax=ax)
    ax.invert_yaxis()

    return ax

def hz_to_midi(freqs):
    return 12 * np.log2(freqs / 440) + 69

def midi_to_hz(midi):
    return 440 * 2  ** ((midi - 69) / 12)

def tonp(tensor):
    return tensor.detach().cpu().numpy()

def stft_to_mel(stft, sr: int, n_fft: int, hop: int):
    return li.feature.melspectrogram(S=stft, sr=sr, n_fft=n_fft, hop_length=hop)

IDX = 0

def reconstruction_report(writer: SummaryWriter, config: dict,
                          original_stft: np.ndarray, reconstructed_stft: np.ndarray,
                          harmonic_amps: np.ndarray, noise_filter: np.ndarray,
                          f0: np.ndarray, loudness: np.ndarray, tag: str, step: int):

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 5))

    # we get a multi-scale spectrogram, but we want to only access 1
    scale_idx = len(config['train']['scales'])//2
    sr = config['preprocess']['sampling_rate']
    n_fft = config['train']['scales'][IDX]
    hop = config['train']['overlap']

    original_stft = tonp(original_stft[IDX][scale_idx])
    original_stft = stft_to_mel(original_stft, sr, n_fft, hop)
    axes[1][0].set_title('Original')
    plot_spec(original_stft, axes[0][0])

    reconstructed_stft = tonp(reconstructed_stft[IDX][scale_idx])
    reconstructed_stft = stft_to_mel(reconstructed_stft, sr, n_fft, hop)
    axes[1][0].set_title('Reconstruction')
    plot_spec(reconstructed_stft, axes[1][0])

    f0 = tonp(f0[IDX].squeeze(-1))
    midi = hz_to_midi(f0)
    axes[0][1].set_title('F0 contour')
    axes[0][1].set_ylim([-1, 128])
    plot_sig(midi, axes[0][1])

    loudness = tonp(loudness[IDX].squeeze(-1))
    axes[1][1].set_title('Loudness')
    plot_sig(midi, axes[1][1])

    noise_filter = tonp(noise_filter[IDX].T)
    axes[0][2].set_title('Noise Filter')
    plot_spec(noise_filter, axes[0][2], amp_to_db=False)

    harmonic_amps = tonp(harmonic_amps[IDX].T)
    axes[1][2].set_title('Harmonic Envelope')
    plot_spec(harmonic_amps, axes[1][2])

    fig.suptitle('reconstruction report')
    fig.tight_layout()

    writer.add_figure(tag, fig, step)
    return fig, axes
