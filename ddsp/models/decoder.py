import ddsp
from ddsp.models.modules import Reverb, HarmonicSynth, FilteredNoise

import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class GRUDecoder(nn.Module):

    def __init__(self, hidden_size: int, z_dim: int = None):
        """ddsp GRU decoder

        Args:
            hidden_size (int): gru hidden size. real hidden size will be 2*hidden_size 
                             without Z and 3*hidden_size with Z
            z_dim (int, optional): dimensionality of Z. If None, no Z is added.  
        """
        super().__init__()
        self.register_buffer("cache_gru", torch.zeros(1, 1, hidden_size))

        N_LAYERS = 3

        # projections from f0 and loudness to GRU
        self.f0_mlp = ddsp.mlp(in_size=1,
                               hidden_size=hidden_size,
                               n_layers=N_LAYERS)
        self.loudness_mlp = ddsp.mlp(in_size=1,
                                     hidden_size=hidden_size,
                                     n_layers=N_LAYERS)

        # add a z projection if needed
        self.add_z = z_dim is not None
        if self.add_z:
            self.z_mlp = ddsp.mlp(z_dim, hidden_size, N_LAYERS)

        # GRU decoder
        # add an extra hidden block if we're adding Z
        num_hiddens = 2 if not self.add_z else 3
        self.gru = ddsp.gru(num_hiddens, hidden_size)
        self.out_mlp = ddsp.mlp(hidden_size + 2, hidden_size, N_LAYERS)

    def forward(self, f0, loudness, z=None, realtime=False):
        hidden = torch.cat([
            self.f0_mlp(f0),
            self.loudness_mlp(loudness),
        ], -1)

        
        print("Concatenated output of f0 and loudness MLPs:", hidden.shape)
        
        # concatenate z if we need to
        if self.add_z:
            assert z is not None
            hidden = torch.cat([
                hidden,
                self.z_mlp(z)
            ], -1)
        
        print("Concatenated output of f0 loudness MLPs with encoded latents:", hidden.shape)

        # TODO: why are we passing f0 and loudness through a skip conn? (here)
        if realtime:
            gru_out, cache = self.gru(hidden, self.cache_gru)
            self.cache_gru.copy_(cache)

            hidden = torch.cat([gru_out, f0, loudness], -1)
            hidden = self.out_mlp(hidden)
        else:
            hidden = torch.cat([self.gru(hidden)[0], f0, loudness], -1)
            
            print("Concatenated output of GRU with pitch and loudness", hidden.shape)
            
            hidden = self.out_mlp(hidden)
            
            print("Final MLP output of decoder", hidden.shape)

        return hidden

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

        # GRU decoder
        self.decoder = GRUDecoder(hidden_size=hidden_size, z_dim=None)

        # projections to harmonics and noise
        self.harmonic_proj = nn.Linear(hidden_size, n_harmonic + 1)
        self.noise_proj = nn.Linear(hidden_size, n_bands)

        # synths
        self.harmonic_synth = HarmonicSynth(block_size=block_size,
                                            sample_rate=sample_rate)
        self.noise_synth = FilteredNoise(block_size=block_size,
                                         window_size=n_bands)

        # reverb
        self.has_reverb = has_reverb
        self.reverb = Reverb(sample_rate, sample_rate)

        self.register_buffer("phase", torch.zeros(1))        

    def forward(self, batch: dict):
        f0, loudness = batch['pitch'], batch['loudness']
        hidden = self.decoder(f0, loudness)

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
        hidden = self.decoder(f0, loudness, realtime=True)

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