import ddsp
from ddsp.modules import Reverb, HarmonicSynth, FilteredNoise

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class DDSPDecoder(nn.Module):
    """
    DDSP GRU Decoder with no encoded Z dimension.

    This model uses only f0 and loudness as input. 
    """
    def __init__(self, hidden_size: int, n_harmonic: int, n_bands: int, sample_rate: int,
                 block_size: int, has_reverb: bool):
        super().__init__()
        self.register_buffer("sample_rate", torch.tensor(sample_rate))
        self.register_buffer("block_size", torch.tensor(block_size))

        # projections from f0 and loudness to GRU
        self.f0_mlp = ddsp.mlp(in_size=1, hidden_size=hidden_size, n_layers=3)
        self.loudness_mlp = ddsp.mlp(in_size=1,
                                     hidden_size=hidden_size,
                                     n_layers=3)

        # GRU decoder
        self.gru = ddsp.gru(2, hidden_size)
        self.out_mlp = ddsp.mlp(hidden_size + 2, hidden_size, 3)

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

    def forward(self, f0, loudness):
        # forward pass through decoder model
        hidden = torch.cat([
            self.f0_mlp(f0),
            self.loudness_mlp(loudness),
        ], -1)
        hidden = torch.cat([self.gru(hidden)[0], f0, loudness], -1)
        hidden = self.out_mlp(hidden)

        # harmonic synth
        param = self.harmonic_proj(hidden)
        amplitudes = param[..., :1]
        harmonic_distribution = param[..., 1:]

        harmonic_ctrls = self.harmonic_synth.get_controls(amplitudes, harmonic_distribution,
                                                             f0)
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
            'f0': f0,
            'loudness': loudness,
            'signal': signal,
            'noise': noise,
            'harmonic_audio': harmonic,
            'noise_ctrls': noise_ctrls,
            'harmonic_ctrls': harmonic_ctrls
        }
        return output

    def realtime_forward(self, f0, loudness):
        # forward pass through decoder model
        hidden = torch.cat([
            self.f0_mlp(f0),
            self.loudness_mlp(loudness),
        ], -1)

        gru_out, cache = self.gru(hidden, self.cache_gru)
        self.cache_gru.copy_(cache)

        hidden = torch.cat([gru_out, f0, loudness], -1)
        hidden = self.out_mlp(hidden)

        # harmonic part
        param = self.proj_matrices[0](hidden)
        amplitudes = param[..., :1]
        harmonic_distribution = param[..., 1:]

        harmonic_ctrls = self.harmonic_synth.get_controls(amplitudes, harmonic_distribution,
                                                          f0)
        harmonic = self.harmonic_synth(**harmonic_ctrls)

        # noise part
        magnitudes = self.proj_matrices[1](hidden)
        noise_ctrls = self.noise_synth.get_controls(magnitudes)
        noise = self.noise_synth(**noise_ctrls)

        signal = harmonic + noise

        return signal

    def reconstruction_report(self, original, reconstructed,
                              config: dict, output: dict, index=0):
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 5))

        scale_idx = len(config['train']['scales'])//2
        sr = config['preprocess']['sample_rate']
        n_fft = config['train']['scales'][index]
        hop = config['train']['overlap']

        original = ddsp.utils.tonp(original[index][scale_idx])
        original = ddsp.utils.stft_to_mel(original, sr, n_fft, hop)
        axes[0][0].set_title('Original')
        ddsp.utils.plot_spec(original, axes[0][0])

        reconstructed = ddsp.utils.tonp(reconstructed[index][scale_idx])
        reconstructed = ddsp.utils.stft_to_mel(reconstructed, sr,
                                                    n_fft, hop)
        axes[1][0].set_title('Reconstruction')
        ddsp.utils.plot_spec(reconstructed, axes[1][0])

        ddsp.utils.plot_f0(axes[0][1], output['f0'], index)
        ddsp.utils.plot_loudness(axes[1][1], output['loudness'], index)

        self.noise_synth.plot(axes[0][2], ctrls=output['noise_ctrls'], index=index)
        self.harmonic_synth.plot(axes[1][2],
                                 ctrls=output['harmonic_ctrls'],
                                 index=index)

        fig.suptitle('reconstruction report')
        fig.tight_layout()

        return fig