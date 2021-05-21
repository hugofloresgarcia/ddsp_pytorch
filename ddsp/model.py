import torch
import torch.nn as nn
from .core import mlp, gru, scale_function, remove_above_nyquist, upsample
from .core import harmonic_synth, amp_to_impulse_response, fft_convolve


class Reverb(nn.Module):
    def __init__(self, length, sample_rate, initial_wet=0, initial_decay=5):
        super().__init__()
        self.length = length
        self.sample_rate = sample_rate

        self.noise = nn.Parameter((torch.rand(length) * 2 - 1).unsqueeze(-1))
        self.decay = nn.Parameter(torch.tensor(float(initial_decay)))
        self.wet = nn.Parameter(torch.tensor(float(initial_wet)))

        t = torch.arange(self.length) / self.sample_rate
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
        amplitudes = scale_function(amplitudes)
        harmonic_distribution = scale_function(harmonic_distribution)

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

class FilteredNoise(nn.Module):

    def __init__(self, block_size: int, window_size: int,
                 initial_bias: int = -5.0):
        super().__init__()
        self.block_size = block_size
        self.window_size = window_size
        self.initial_bias = initial_bias

    def get_controls(self, magnitudes):
        return {'magnitudes': scale_function(magnitudes + self.initial_bias)}

    def forward(self, magnitudes):
        impulse = amp_to_impulse_response(magnitudes, self.block_size)
        # create a noise vector N(0, 1)
        noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            self.block_size,
        ).to(impulse) * 2 - 1

        noise = fft_convolve(noise, impulse).contiguous()
        noise = noise.reshape(noise.shape[0], -1, 1)

        return noise

class DDSPDecoder(nn.Module):
    """
    DDSP GRU Decoder with no encoded Z dimension.

    This model uses only pitch and loudness as input. 
    """
    def __init__(self, hidden_size: int, n_harmonic: int, n_bands: int, sample_rate: int,
                 block_size: int, has_reverb: bool):
        super().__init__()
        self.register_buffer("sample_rate", torch.tensor(sample_rate))
        self.register_buffer("block_size", torch.tensor(block_size))

        # projections from f0 and loudness to GRU
        self.f0_mlp = mlp(in_size=1, hidden_size=hidden_size, n_layers=3)
        self.loudness_mlp = mlp(in_size=1, hidden_size=hidden_size, n_layers=3)

        # GRU decoder
        self.gru = gru(2, hidden_size)
        self.out_mlp = mlp(hidden_size + 2, hidden_size, 3)

        self.harmonic_proj = nn.Linear(hidden_size, n_harmonic + 1)
        self.noise_proj = nn.Linear(hidden_size, n_bands)

        self.harmonic_synth = HarmonicSynth(block_size=block_size,
                                            sample_rate=sample_rate)
        self.noise_synth = FilteredNoise(block_size=block_size,
                                         window_size=n_bands)

        self.has_reverb = has_reverb
        self.reverb = Reverb(sample_rate, sample_rate)

        self.register_buffer("cache_gru", torch.zeros(1, 1, hidden_size))
        self.register_buffer("phase", torch.zeros(1))

    def forward(self, pitch, loudness):
        # forward pass through decoder model
        hidden = torch.cat([
            self.f0_mlp(pitch),
            self.loudness_mlp(loudness),
        ], -1)
        hidden = torch.cat([self.gru(hidden)[0], pitch, loudness], -1)
        hidden = self.out_mlp(hidden)

        # harmonic synth
        param = self.harmonic_proj(hidden)
        amplitudes = param[..., :1]
        harmonic_distribution = param[..., 1:]

        harmonic_ctrls = self.harmonic_synth.get_controls(amplitudes, harmonic_distribution,
                                                             pitch)
        harmonic = self.harmonic_synth(**harmonic_ctrls)

        # filtered noise
        magnitudes = self.noise_proj(hidden)
        noise_ctrls = self.noise_synth.get_controls(magnitudes)
        noise = self.noise_synth(**noise_ctrls)

        # add signals
        # question: would it make sense to make this a learnable weighted sum?
        signal = harmonic + noise

        # add reverb
        if self.has_reverb:
            signal = self.reverb(signal)

        output = {
            'signal': signal,
            'noise': noise,
            'harmonic_audio': harmonic,
            'noise_magnitudes': noise_ctrls['magnitudes'],
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
        param = self.proj_matrices[0](hidden)
        amplitudes = param[..., :1]
        harmonic_distribution = param[..., 1:]

        harmonic_ctrls = self.harmonic_synth.get_controls(amplitudes, harmonic_distribution,
                                                          pitch)
        harmonic = self.harmonic_synth(**harmonic_ctrls)

        # noise part
        magnitudes = self.proj_matrices[1](hidden)
        noise_ctrls = self.noise_synth.get_controls(magnitudes)
        noise = self.noise_synth(**noise_ctrls)

        signal = harmonic + noise

        return signal