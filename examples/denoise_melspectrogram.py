import glob
import numpy as np

from bm3d import bm3d, BM3DProfile
from file_utils import *


# bm3d parameters
smoothing_factor = 2.5e-1  # Smoothing factor

profile = BM3DProfile()
profile.denoise_residual = True
# profile.gamma = 8.0
# profile.search_window_ht = 101
# profile.search_window_wiener = 101

parameters = f"noise_variance:{smoothing_factor}\n"
for key, value in profile.__dict__.items():
    parameters += f"{key}:{value}\n"

explanation = f"""
1. Scaling mel-spectrogram using log function (10*np.log10(power_sp))
2. Smooth log mel-spectrogram with Block mathing 3D
3. Rescaling log mel-spectrogram (np.power(10.0, 0.1 * db_est))
[bm3d parameters]
{parameters}
"""

def bm3d_denoise(fn, save_audio, plot):
    power_sp = np.load(fn)
    db = np.log10(power_sp)
    z = np.atleast_3d(db)

    db_est = bm3d(z, np.sqrt(smoothing_factor), profile=profile)
    save_files(db, db_est, fn, power_sp,
               save_audio=save_audio, plot=plot)

def main():
    for dir in "../audio ../submit ../fig ../power_spectrum".split():
        os.makedirs(dir, exist_ok=True)

    root_dir = "../dist-data/noised_tgt"
    for file in glob.glob(f"{root_dir}/*.npy"):
        bm3d_denoise(file, save_audio=True, plot=True)

    with open("../submit/readme.txt", "w") as f:
        f.write(explanation)

    save_to_zip()


if __name__ == '__main__':
    main()