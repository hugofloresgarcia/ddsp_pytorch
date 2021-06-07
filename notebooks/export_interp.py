import ddsp
from pathlib import Path
import os

os.getcwd()
os.chdir('/home/hugo/lab/ddsp_pytorch/')

config1 = ddsp.export.load_config('./runs/violin-interp-v1/config.yaml')
config2 = ddsp.export.load_config('./runs/reed_acoustic_011/config.yaml')
out_dir = './exports/reed-violin-interp-v1/'

ddsp.export.export_multidecoder_interpolator(config1, config2, out_dir)