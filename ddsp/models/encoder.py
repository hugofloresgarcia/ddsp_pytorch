import ddsp
from ddsp.models.decoder import GRUDecoder
from ddsp.models.modules import Reverb, HarmonicSynth, FilteredNoise

import librosa
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class MFCCEncoder(nn.Module):

    def __init__(self, sample_rate: int, block_size: int,
                 hidden_size: int, n_mfccs: int, z_dim: int = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.z_dim = z_dim

        self.norm = nn.LayerNorm(n_mfccs)
        self.gru = nn.GRU(n_mfccs, hidden_size, batch_first=True)
        self.proj = nn.Linear(hidden_size, z_dim)


    def forward(self, mfccs: torch.Tensor):
        x = self.norm(mfccs)
        x, _ = self.gru(x)
        z = self.proj(x)
        return z


class StationaryMFCCEncoder(nn.Module):
    """
    Performs a projection of only the first MFCC frame of a signal and tiles it to match sequence length

    """
    def __init__(self, sample_rate: int, block_size: int,
                 hidden_size: int, n_mfccs: int, z_dim: int = None, encoding="average"):
        """

        :param sample_rate:
        :param block_size:
        :param hidden_size:
        :param n_mfccs:
        :param z_dim:
        :param encoding: if 'first', simply tile first MFCC frame. If 'average', tile average of MFCC frames. If 'gram', return flattened gram matrices of a random convolutional layer
        """

        super().__init__()
        self.hidden_size = hidden_size
        self.z_dim = z_dim

        self.norm = nn.LayerNorm(n_mfccs)
        self.proj1 = nn.Linear(n_mfccs, hidden_size)
        self.proj2 = nn.Linear(hidden_size, z_dim)
        self.encoding = encoding

        self.relu = torch.nn.ReLU(inplace=False)  # nonlinearity

        if self.encoding == "gram":

            filter_sizes = [(11, 3), (5, 3), (3, 5), (3, 3), (5, 5), (7, 7), (13, 13)]
            out_channels = 64

            self.conv = torch.nn.ModuleList()
            for s in filter_sizes:
                self.conv.append(torch.nn.Conv2d(in_channels=1,
                                                 out_channels=out_channels,
                                                 kernel_size=s,
                                                 bias=False,
                                                 padding=0)
                                 )
            # initialize weights
            for c in self.conv:
                c.weight.data.uniform_(-0.05, 0.05)

            # do not update convolutional filters
            for p in self.conv.parameters():
                p.requires_grad = False

            self.C = 10

            self.gram_proj1 = nn.Linear(n_mfccs * len(self.conv) * out_channels * out_channels, hidden_size)
            self.gram_proj2 = nn.Linear(hidden_size, z_dim)

    def gram_matrix(self, activations: torch.Tensor):
        """
        :param activations: concatenated feature maps from convolution layer)
        """
        n_batch = activations.shape[0]
        N = activations.shape[1]
        fm_size = torch.tensor(activations.shape[2:])
        M = torch.prod(fm_size)

        F = activations.reshape(n_batch, N, *fm_size).transpose(1, 2)
        G = (F @ F.transpose(2, 3)) / M

        return G

    def forward(self, mfccs: torch.Tensor, length=None):
        n_batch, n_frames, n_mfcc = mfccs.shape

        if self.encoding == "first":
            x = self.norm(mfccs[:, 0, :])   # idea 1: take only take first mfcc frame per signal
            hidden = self.proj1(x)
            hidden = self.relu(hidden)
            z = self.proj2(hidden).reshape(n_batch, 1, self.z_dim)
        elif self.encoding == "average":
            x = self.norm(torch.mean(mfccs, dim=1))  # idea 2: take average over sequence
            hidden = self.proj1(x)
            hidden = self.relu(hidden)
            z = self.proj2(hidden).reshape(n_batch, 1, self.z_dim)
        elif self.encoding == "gram":  # idea 3: texture embedding

            # apply compressive nonlinearity
            x = 2 * torch.sigmoid(self.C * mfccs) - 1  # shape: (n_batch, n_frames, n_mfcc)
            # x = torch.log(1 + self.C * x) / torch.log(1 + self.C)  # shape: (n_batch, n_frames, n_mfcc)
            x = torch.unsqueeze(x, 1)  # shape: (n_batch, 1, n_frames, n_mfcc)

            x = x.transpose(2, 3)  # shape: (n_batch, 1, n_mfcc, n_frames)

            # apply convolutional filters in parallel and concatenate
            feature_maps = []
            for i, c in enumerate(self.conv):
                feature_maps.append(self.relu(c(x)))  # shape: (n_batch, out_channels, ~n_mfcc, ~n_frames)

            # compute a Gram correlation matrix per feature map set, pad along frequency dimension, flatten and store
            gram_matrices = []
            for f in feature_maps:
                g = self.gram_matrix(f)

                g = nn.functional.pad(g, pad=(
                0, 0, 0, 0, 0, x.shape[2] - g.shape[1]))  # pad along frequency dimension before concatenating

                gram_matrices.append(g.reshape((g.shape[0], -1)))  # preserve batch dimension

            x = torch.stack(gram_matrices, dim=1)  # shape: (n_batch, n_filter_groups, flattened_gram_length)

            x = x.reshape(n_batch, -1)
            x = self.gram_proj1(x)
            x = self.relu(x)
            z = self.gram_proj2(x)
            z = z.reshape(n_batch, 1, self.z_dim)


        else:  # idea 4: vae
            pass

        if length:
            z = z.repeat(1, length, 1)  # tile to sequence length
        else:
            z = z.repeat(1, n_frames, 1)  # tile to sequence length

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


class DDSPTimbreOnlyEncoder(nn.Module):

    """
    DDSP GRU with deterministic Autoencoder; pitch/loudness are only fed to decoder, leading to less effective
    timbre disentanglement
    """
    def __init__(self, hidden_size: int, n_harmonic: int, n_bands: int,
                 sample_rate: int, block_size: int, has_reverb: bool):
        super().__init__()
        self.register_buffer("sample_rate", torch.tensor(sample_rate))
        self.register_buffer("block_size", torch.tensor(block_size))

        # Timbre-only GRU encoder
        self.encoder = MFCCEncoder(sample_rate, block_size, hidden_size, n_mfccs=30,
                                   z_dim=16)

        # Timbre-only GRU decoder
        self.z_mlp = ddsp.mlp(16, hidden_size, 3)
        self.gru = ddsp.gru(1, hidden_size)
        self.out_mlp = ddsp.mlp(hidden_size + 2, hidden_size, 3)

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

        # project latent to hidden dimension
        hidden = self.z_mlp(z)

        # pass through GRU
        hidden = torch.cat([self.gru(hidden)[0], f0, loudness], -1)
        hidden = self.out_mlp(hidden)

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


class DDSPInterpolator(nn.Module):

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def sample(self, source: dict, ref1: dict, ref2: dict, alpha: torch.Tensor):

        # obtain required dimensions to match source signal
        dim_test = self.model.encode(source)

        n_frames = dim_test.shape[1]

        print(n_frames)

        z1 = self.model.encode(ref1, length=n_frames)
        z2 = self.model.encode(ref2, length=n_frames)

        print(z1.shape, z2.shape)

        for i, d in enumerate(dim_test.shape):
            z1 = z1.narrow(i, 0, d)
            z2 = z2.narrow(i, 0, d)

        print(dim_test.shape, z1.shape, z2.shape)

        # fade between encodings according to schedule parameter alpha
        z = z1 * alpha + z2 * (1 - alpha)
        output = self.model.decode(z, source)

        return output



class DDSPStationaryTimbreEncoder(nn.Module):

    """
    DDSP GRU with deterministic Autoencoder, but with a non-time-varying timbre representation
    """
    def __init__(self, hidden_size: int, n_harmonic: int, n_bands: int,
                 sample_rate: int, block_size: int, has_reverb: bool,
                 encoding="average"):
        super().__init__()
        self.register_buffer("sample_rate", torch.tensor(sample_rate))
        self.register_buffer("block_size", torch.tensor(block_size))

        # GRU encoder
        self.encoder = StationaryMFCCEncoder(sample_rate, block_size, hidden_size, n_mfccs=30, z_dim=16, encoding=encoding)

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

    def encode(self, batch: dict, length=None):
        f0, loudness, mfcc = batch['pitch'], batch['loudness'], batch['mfcc']

        # get the latent
        z = self.encoder(mfcc, length=length)

        return z

    def decode(self, z: torch.Tensor, batch: dict):
        f0, loudness, mfcc = batch['pitch'], batch['loudness'], batch['mfcc']

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

    def forward(self, batch: dict):

        # get the latent
        z = self.encode(batch)

        output = self.decode(z, batch)

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



class DDSPVRNN(nn.Module):

    """
    DDSP with Variational Recurrent Autoencoder.
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