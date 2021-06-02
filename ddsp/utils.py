import numpy as np
import matplotlib.pyplot as plt
import librosa as li
import torch

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

def lin_interp(a: torch.Tensor, b: torch.Tensor, alpha: torch.Tensor):
    return alpha * a + (1 - alpha) * b

def hz_to_midi(freqs):
    return 12 * np.log2(freqs / 440) + 69

def midi_to_hz(midi):
    return 440 * 2  ** ((midi - 69) / 12)

def tonp(tensor):
    return tensor.detach().cpu().numpy()

def stft_to_mel(stft, sr: int, n_fft: int, hop: int):
    return li.feature.melspectrogram(S=stft, sr=sr, n_fft=n_fft, hop_length=hop)

def plot_f0(ax, f0, index: int = 0):
    f0 = tonp(f0[index].squeeze(-1))
    midi = hz_to_midi(f0)
    ax.set_title('F0 (MIDI)')
    ax.set_ylim([-1, 128])
    plot_sig(midi, ax)

def plot_loudness(ax, loudness, index: int = 0):
    loudness = tonp(loudness[index].squeeze(-1))
    ax.set_title('Loudness')
    plot_sig(loudness, ax)

def log_step(model, writer, output: dict, stage: str, step: int, config: dict):
    IDX = 0

    writer.add_scalar(f'loss/{stage}', output['loss'].item(), step)
    writer.add_scalar("reverb_decay", model.reverb.decay.item(), step)
    writer.add_scalar("reverb_wet", model.reverb.wet.item(), step)

    # log the audio to tb (instead of writing to file)
    writer.add_audio(f'sig/{stage}',
                        output['sig'][IDX],
                        global_step=step,
                        sample_rate=config['preprocess']["sample_rate"])
    writer.add_audio(f'rec/{stage}',
                        output['rec'][IDX],
                        global_step=step,
                        sample_rate=config['preprocess']["sample_rate"])

    fig = model.reconstruction_report(output['sig_stft'],
                                      output['rec_stft'],
                                      config,
                                      output,
                                      index=IDX)

    writer.add_figure(f'reconstruction/{stage}', fig, step)