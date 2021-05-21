import torch
import torch.nn as nn
from .core import mlp, gru, scale_function, remove_above_nyquist, upsample
from .core import harmonic_synth, amp_to_impulse_response, fft_convolve
from .core import resample
import math


class Reverb(nn.Module):
    def __init__(self, length, sampling_rate, initial_wet=0, initial_decay=5):
        super().__init__()
        self.length = length
        self.sampling_rate = sampling_rate

        self.noise = nn.Parameter((torch.rand(length) * 2 - 1).unsqueeze(-1))
        self.decay = nn.Parameter(torch.tensor(float(initial_decay)))
        self.wet = nn.Parameter(torch.tensor(float(initial_wet)))

        t = torch.arange(self.length) / self.sampling_rate
        t = t.reshape(1, -1, 1)
        self.register_buffer("t", t)

    def build_impulse(self):
        t = torch.exp(-nn.functional.softplus(-self.decay) * self.t * 500)
        noise = self.noise * t
        impulse = noise * torch.sigmoid(self.wet)
        impulse[:, 0] = 1
        return impulse

    def forward(self, x):
        lenx = x.shape[1]
        impulse = self.build_impulse()
        impulse = nn.functional.pad(impulse, (0, 0, 0, lenx - self.length))

        x = fft_convolve(x.squeeze(-1), impulse.squeeze(-1)).unsqueeze(-1)

        return x

class HarmonicSynth(nn.Module):

    def __init__(self, block_size: int,
                 sample_rate: int):
        super().__init__()
        self.block_size = block_size
        self.sample_rate = sample_rate

    def get_controls(self, amplitudes, harmonic_distribution, f0):
        """gets control parameters for the synthesizer.

        Args:
            amplitudes: frame-wise amplitudes (batch, frame, 1)
            harmonic_distribution: frame-wise harmonic distribution 
                                    (batch, frame, harmonic)
            f0: fundamental frequency
        """

        harmonic_distribution = remove_above_nyquist(
            harmonic_distribution,
            f0,
            self.sample_rate,
        )
        harmonic_distribution /= harmonic_distribution.sum(-1, keepdim=True)

        return {
                'f0': f0,
                'harmonic_distribution': harmonic_distribution,
                'amplitudes': amplitudes
            }

    def forward(self, amplitudes, harmonic_distribution, f0):
        """ forward pass through synthesizer 
        (must run get_controls first)
        """
        harmonic_distribution *= amplitudes
        harmonic_distribution = upsample(harmonic_distribution,
                                         self.block_size)
        f0 = upsample(f0, self.block_size)

        sig = harmonic_synth(f0, harmonic_distribution, self.sample_rate)

        return sig

class DDSP(nn.Module):
    """
    DDSP Decoder with no encoded Z dimension.

    This model uses only pitch and loudness as input. 
    """
    def __init__(self, hidden_size: int, n_harmonic: int, n_bands: int, sampling_rate: int,
                 block_size: int, has_reverb: bool):
        super().__init__()
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))

        self.in_mlps = nn.ModuleList([mlp(1, hidden_size, 3)] * 2)
        self.gru = gru(2, hidden_size)
        self.out_mlp = mlp(hidden_size + 2, hidden_size, 3)

        self.proj_matrices = nn.ModuleList([
            nn.Linear(hidden_size, n_harmonic + 1),
            nn.Linear(hidden_size, n_bands),
        ])

        self.harmonic_synth = HarmonicSynth(block_size=block_size,
                                            sample_rate=sampling_rate)

        self.has_reverb = has_reverb
        self.reverb = Reverb(sampling_rate, sampling_rate)

        self.register_buffer("cache_gru", torch.zeros(1, 1, hidden_size))
        self.register_buffer("phase", torch.zeros(1))

    def forward(self, pitch, loudness):
        hidden = torch.cat([
            self.in_mlps[0](pitch),
            self.in_mlps[1](loudness),
        ], -1)
        hidden = torch.cat([self.gru(hidden)[0], pitch, loudness], -1)
        hidden = self.out_mlp(hidden)

        # harmonic part
        # breakpoint()
        param = scale_function(self.proj_matrices[0](hidden))

        amplitudes = param[..., :1]
        harmonic_distribution = param[..., 1:]

        harmonic_ctrls = self.harmonic_synth.get_controls(amplitudes, harmonic_distribution,
                                                             pitch)
        harmonic = self.harmonic_synth(**harmonic_ctrls)

        # noise part
        param = scale_function(self.proj_matrices[1](hidden) - 5)

        impulse = amp_to_impulse_response(param, self.block_size)
        # create a noise vector N(0, 1)
        noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            self.block_size,
        ).to(impulse) * 2 - 1

        noise = fft_convolve(noise, impulse).contiguous()
        noise = noise.reshape(noise.shape[0], -1, 1)

        signal = harmonic + noise

        #reverb part
        if self.has_reverb:
            signal = self.reverb(signal)

        output = {
            'signal': signal,
            'noise': noise,
            'harmonic_audio': harmonic,
            'noise_filter': torch.fft.rfft(impulse).abs(),
            'reverb_impulse': self.reverb.build_impulse(),
        }

        output.update(harmonic_ctrls)

        return output

    def realtime_forward(self, pitch, loudness):
        hidden = torch.cat([
            self.in_mlps[0](pitch),
            self.in_mlps[1](loudness),
        ], -1)

        gru_out, cache = self.gru(hidden, self.cache_gru)
        self.cache_gru.copy_(cache)

        hidden = torch.cat([gru_out, pitch, loudness], -1)
        hidden = self.out_mlp(hidden)

        # harmonic part
        param = scale_function(self.proj_matrices[0](hidden))

        amplitudes = param[..., :1]
        harmonic_distribution = param[..., 1:]

        harmonic_ctrls = self.harmonic_synth.get_controls(amplitudes, harmonic_distribution, pitch)
        harmonic = self.harmonic_synth(**harmonic_ctrls)

        # noise part
        param = scale_function(self.proj_matrices[1](hidden) - 5)

        impulse = amp_to_impulse_response(param, self.block_size)
        noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            self.block_size,
        ).to(impulse) * 2 - 1

        noise = fft_convolve(noise, impulse).contiguous()
        noise = noise.reshape(noise.shape[0], -1, 1)

        signal = harmonic + noise

        return signal