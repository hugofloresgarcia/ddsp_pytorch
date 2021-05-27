import ddsp
from ddsp.models.decoder import GRUDecoder
from ddsp.models.modules import Reverb, HarmonicSynth, FilteredNoise

import librosa
import torch
import torch.nn as nn

class MFCCEncoder(nn.Module):

    def __init__(self, sample_rate: int, block_size: int,
                 hidden_size: int, n_mfccs: int, z_dim: int = None):
        self.hidden_size = hidden_size
        self.z_dim = z_dim

        self.norm = nn.LayerNorm(n_mfccs)
        self.gru = ddsp.gru(n_mfccs, hidden_size)
        self.proj = nn.Linear(hidden_size, z_dim)


    def forward(self, mfccs: torch.Tensor):
        x = self.norm(mfccs)
        x, _ = self.gru(x)
        z = self.proj(x)
        breakpoint()
        return z

class DDSPAutoencoder(nn.Module):

    """
    DDSP GRU with deterministic Autoencoder. 
    """
    def __init__(self, hidden_size: int, n_harmonic: int, n_bands: int,
                 sample_rate: int, block_size: int, has_reverb: bool):
        super().__init__()
        self.register_buffer("sample_rate", torch.tensor(sample_rate))
        self.register_buffer("block_size", torch.tensor(block_size))

        # GRU encoder
        self.encoder = MFCCEncoder(sample_rate, block_size, hidden_size, n_mfccs=30, 
                                   z_dim=16)

        # GRU decoder
        self.decoder = GRUDecoder(hidden_size=hidden_size, z_dim=16)

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
        f0, loudness, mfcc = batch['pitch'], batch['loudness'], batch['mfcc']

        # get the latent
        z = self.encoder(mfcc)

        hidden = self.decoder(f0, loudness, z=z)

        # harmonic synth
        param = self.harmonic_proj(hidden)
        amplitudes = param[..., :1]
        harmonic_distribution = param[..., 1:]

        harmonic_ctrls = self.harmonic_synth.get_controls(
            amplitudes, harmonic_distribution, f0)
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
            'harmonic_ctrls': harmonic_ctrls,
            'z': z,
        }
        return output

    def reconstruction_report(self,
                              original,
                              reconstructed,
                              config: dict,
                              output: dict,
                              index=0):
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 5))

        scale_idx = len(config['train']['scales']) // 2
        sr = config['preprocess']['sample_rate']
        n_fft = config['train']['scales'][index]
        hop = config['train']['overlap']

        original = ddsp.utils.tonp(original[index][scale_idx])
        original = ddsp.utils.stft_to_mel(original, sr, n_fft, hop)
        axes[0][0].set_title('Original')
        ddsp.utils.plot_spec(original, axes[0][0])

        reconstructed = ddsp.utils.tonp(reconstructed[index][scale_idx])
        reconstructed = ddsp.utils.stft_to_mel(reconstructed, sr, n_fft, hop)
        axes[1][0].set_title('Reconstruction')
        ddsp.utils.plot_spec(reconstructed, axes[1][0])

        ddsp.utils.plot_f0(axes[0][1], output['f0'], index)
        ddsp.utils.plot_loudness(axes[1][1], output['loudness'], index)

        self.noise_synth.plot(axes[0][2],
                              ctrls=output['noise_ctrls'],
                              index=index)
        self.harmonic_synth.plot(axes[1][2],
                                 ctrls=output['harmonic_ctrls'],
                                 index=index)

        fig.suptitle('reconstruction report')
        fig.tight_layout()

        return fig