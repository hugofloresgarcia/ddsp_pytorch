import numpy as np
import matplotlib.pyplot as plt
import librosa as li

def get_scheduler(len_dataset, start_lr, stop_lr, length):
    def schedule(epoch):
        step = epoch * len_dataset
        if step < length:
            t = step / length
            return start_lr * (1 - t) + stop_lr * t
        else:
            return stop_lr

    return schedule


def plot_spec(stft):
    """ returns a fig and an ax"""
    plt.close('all')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 7))
    pr = lambda m: li.amplitude_to_db(m)
    ax.imshow(pr(stft), aspect='auto')
    ax.invert_yaxis()

    return fig, ax