import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np
import librosa as li
import crepe
import math


def safe_log(x):
    return torch.log(x + 1e-7)


@torch.no_grad()
def mean_std_loudness(dataset):
    mean = 0
    std = 0
    n = 0
    for batch in dataset:
        l = batch['loudness']
        n += 1
        mean += (l.mean().item() - mean) / n
        std += (l.std().item() - std) / n
    return mean, std


def multiscale_fft(signal, scales, overlap):
    stfts = []
    for s in scales:
        S = torch.stft(
            signal,
            s,
            int(s * (1 - overlap)),
            s,
            torch.hann_window(s).to(signal),
            True,
            normalized=True,
            return_complex=True,
        ).abs()
        stfts.append(S)
    return stfts


def resample(x, factor: int):
    batch, frame, channel = x.shape
    x = x.permute(0, 2, 1).reshape(batch * channel, 1, frame)

    window = torch.hann_window(
        factor * 2,
        dtype=x.dtype,
        device=x.device,
    ).reshape(1, 1, -1)
    y = torch.zeros(x.shape[0], x.shape[1], factor * x.shape[2]).to(x)
    y[..., ::factor] = x
    y[..., -1:] = x[..., -1:]
    y = torch.nn.functional.pad(y, [factor, factor])
    y = torch.nn.functional.conv1d(y, window)[..., :-1]

    y = y.reshape(batch, channel, factor * frame).permute(0, 2, 1)

    return y


def upsample(signal, factor):
    signal = signal.permute(0, 2, 1)
    signal = nn.functional.interpolate(signal, size=signal.shape[-1] * factor)
    return signal.permute(0, 2, 1)


def remove_above_nyquist(amplitudes, f0, sample_rate):
    n_harm = amplitudes.shape[-1]
    pitches = f0 * torch.arange(1, n_harm + 1).to(f0)
    aa = (pitches < sample_rate / 2).float() + 1e-4
    return amplitudes * aa


def scale_function(x):
    return 2 * torch.sigmoid(x)**(math.log(10)) + 1e-7


def extract_loudness(signal, sample_rate, block_size, n_fft=2048):
    S = li.stft(
        signal,
        n_fft=n_fft,
        hop_length=block_size,
        win_length=n_fft,
        center=True,
    )
    S = np.log(abs(S) + 1e-7)
    f = li.fft_frequencies(sample_rate, n_fft)
    a_weight = li.A_weighting(f)

    S = S + a_weight.reshape(-1, 1)

    S = np.mean(S, 0)[..., :-1]

    return S


def extract_pitch(signal, sample_rate, block_size):
    length = signal.shape[-1] // block_size
    f0 = crepe.predict(
        signal,
        sample_rate,
        step_size=int(1000 * block_size / sample_rate),
        verbose=1,
        center=True,
        viterbi=True,
    )
    f0 = f0[1].reshape(-1)[:-1]

    if f0.shape[-1] != length:
        f0 = np.interp(
            np.linspace(0, 1, length, endpoint=False),
            np.linspace(0, 1, f0.shape[-1], endpoint=False),
            f0,
        )

    return f0


def mlp(in_size, hidden_size, n_layers):
    channels = [in_size] + (n_layers) * [hidden_size]
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(channels[i], channels[i + 1]))
        net.append(nn.LayerNorm(channels[i + 1]))
        net.append(nn.LeakyReLU())
    return nn.Sequential(*net)


def gru(n_input, hidden_size):
    return nn.GRU(n_input * hidden_size, hidden_size, batch_first=True)


def harmonic_synth(f0, amplitudes, sample_rate):
    n_harmonic = amplitudes.shape[-1]
    omega = torch.cumsum(2 * math.pi * f0 / sample_rate, 1)
    omegas = omega * torch.arange(1, n_harmonic + 1).to(omega)
    signal = (torch.sin(omegas) * amplitudes).sum(-1, keepdim=True)
    return signal


def amp_to_impulse_response(amp, target_size):
    """ 
    converts a frequency-domain filter response
    into a time-domain impulse response?
    """ 
    # convert to complex  
    amp = torch.stack([amp, torch.zeros_like(amp)], -1)
    amp = torch.view_as_complex(amp)

    # from freq domain to time domain
    impulse = fft.irfft(amp)

    filter_size = impulse.shape[-1]

    impulse = torch.roll(impulse, filter_size // 2, -1)
    win = torch.hann_window(filter_size, dtype=impulse.dtype, device=impulse.device)

    impulse = impulse * win

    impulse = nn.functional.pad(impulse, (0, int(target_size) - int(filter_size)))
    impulse = torch.roll(impulse, -filter_size // 2, -1)

    return impulse


def fft_convolve(signal, kernel):
    signal = nn.functional.pad(signal, (0, signal.shape[-1]))
    kernel = nn.functional.pad(kernel, (kernel.shape[-1], 0))

    output = fft.irfft(fft.rfft(signal) * fft.rfft(kernel))
    output = output[..., output.shape[-1] // 2:]

    return output