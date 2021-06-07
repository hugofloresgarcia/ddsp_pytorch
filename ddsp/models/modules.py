import ddsp

import torch
import torch.nn as nn

def interpolate_controls(interp_keys, ctrls1, ctrls2, alpha):
    """ will interpolate the keys in interp_keys. 
    any keys not in interp_keys will default to the ones in ctrls1
    """
    interp_ctrls = {}
    for key in interp_keys:
        interp_ctrls[key] = ddsp.utils.lin_interp(
            ctrls1[key],
            ctrls2[key],
            alpha
        )
    
    out_ctrls = dict(ctrls1)
    out_ctrls.update(interp_ctrls)
    
    return out_ctrls

class Reverb(nn.Module):
    def __init__(self, length, sample_rate, initial_wet=0, initial_decay=5):
        super().__init__()

        self.length = length
        self.register_buffer("sample_rate", torch.tensor(sample_rate))

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

        x = ddsp.fft_convolve(x.squeeze(-1), impulse.squeeze(-1)).unsqueeze(-1)

        return x

class HarmonicSynth(nn.Module):
    def __init__(self, block_size: int, sample_rate: int):
        super().__init__()
        self.register_buffer("sample_rate", torch.tensor(sample_rate))
        self.register_buffer("block_size", torch.tensor(block_size))

    def get_controls(self, amplitudes, harmonic_distribution, f0):
        """gets control parameters for the synthesizer.

        Args:
            amplitudes: frame-wise amplitudes (batch, frame, 1)
            harmonic_distribution: frame-wise harmonic distribution 
                                    (batch, frame, harmonic)
            f0: fundamental frequency
        """
        amplitudes = ddsp.scale_function(amplitudes)
        harmonic_distribution = ddsp.scale_function(harmonic_distribution)

        harmonic_distribution = ddsp.remove_above_nyquist(
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
        harmonic_distribution = ddsp.upsample(harmonic_distribution,
                                              self.block_size)
        f0 = ddsp.upsample(f0, self.block_size)

        sig = ddsp.harmonic_synth(f0, harmonic_distribution, self.sample_rate)

        return sig

    def plot(self, ax, ctrls: dict, index: int = 0):
        """plot harmonic distribution onto an ax

        Args:
            ax : matplotlib ax
            ctrls (dict): dict provided by get_controls
            index (int, optional): batch index to plot. Defaults to 0.
        """
        harmonic_distribution = ddsp.utils.tonp(
            ctrls['harmonic_distribution'])[index].T
        ax.set_title('harmonic distribution')
        ax.set_xlabel('frames')
        ax.set_ylabel('harmonic number')

        ddsp.utils.plot_spec(harmonic_distribution, ax, amp_to_db=True)

        return ax


class FilteredNoise(nn.Module):
    def __init__(self,
                 block_size: int,
                 window_size: int,
                 initial_bias: int = -5.0):
        super().__init__()
        self.register_buffer("block_size", torch.tensor(block_size))
        self.window_size = window_size
        self.initial_bias = initial_bias

    def get_controls(self, magnitudes):
        return {
            'magnitudes': ddsp.scale_function(magnitudes + self.initial_bias)
        }

    def forward(self, magnitudes):
        impulse = ddsp.amp_to_impulse_response(magnitudes, int(self.block_size))
        # create a noise vector N(0, 1)
        noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            self.block_size,
        ).to(impulse) * 2 - 1

        noise = ddsp.fft_convolve(noise, impulse).contiguous()
        noise = noise.reshape(noise.shape[0], -1, 1)

        return noise

    def plot(self, ax, ctrls: dict, index: int = 0):
        """plot noise magnitudes (in freq domain) onto an ax

        Args:
            ax : matplotlib ax
            ctrls (dict): dict provided by get_controls
            index (int, optional): batch index to plot. Defaults to 0.
        """
        magnitudes = ddsp.utils.tonp(ctrls['magnitudes'])[index].T

        ax.set_title('noise magnitude')
        ax.set_xlabel('frames')
        ax.set_ylabel('frequency bin')

        ddsp.utils.plot_spec(magnitudes, ax, amp_to_db=True)

        return ax
