"""
Grayscale BM3D denoising demo file, based on Y. Mäkinen, L. Azzari, A. Foi, 2019.
Exact Transform-Domain Noise Variance for Collaborative Filtering of Stationary Correlated Noise.
In IEEE International Conference on Image Processing (ICIP), pp. 185-189
"""
import os
import glob
import shutil
import numpy as np
import librosa
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.metrics import mean_squared_error
from bm3d import bm3d, BM3DProfile
from experiment_funcs import get_experiment_noise, get_psnr, get_cropped_psnr


# bm3d parameters
smoothing_factor = 3.e-3  # Smoothing factor
seed = 0  # seed for pseudorandom noise realization
sr = 22050

profile = BM3DProfile()
profile.denoise_residual = True
# profile.gamma = 8.0
# profile.search_window_ht = 101
# profile.search_window_wiener = 101

parameters = f"noise_variance:{smoothing_factor}\n"
for key, value in profile.__dict__.items():
    parameters += f"{key}:{value}\n"


def power_to_db(power_sp):
    db = 10 * np.log10(power_sp)
    db_abs_max = np.max(np.abs(db))
    db /= db_abs_max
    return db, db_abs_max

def fix_denoised_db(db_denoised, db_abs_max):
    db_denoised *= db_abs_max
    return db_denoised

def plot_db(db, db_denoised, fn):
    fig, axes = plt.subplots(2, 1)
    for ax, im in zip(axes, [db, db_denoised]):
        ax.imshow(im)
    output_fn = os.path.basename(fn).replace(".npy", ".png")
    output_fig = f"../fig/{output_fn}"
    plt.savefig(output_fig)
    plt.clf()
    plt.close()

def plot_3d(spectrogram):
    x = np.arange(spectrogram.shape[1])
    y = np.arange(spectrogram.shape[0])
    xx, yy = np.meshgrid(x, y)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("freq")
    ax.set_zlabel("f(x, y)")

    ax.plot_surface(xx, yy, spectrogram, cmap='viridis')
    plt.show()

def histogram(signal):
    plt.hist(signal)
    plt.show()
    plt.close()
    plt.clf()

def write_wav(audio, fn):
    new_fn = os.path.basename(fn).replace("noised_", "").replace(".npy", ".wav")
    output_fn = f"../audio/{new_fn}"
    librosa.output.write_wav(output_fn, audio, sr=sr)

def calc_mse(power_sp, power_sp_est, fn):
    assert power_sp.shape == power_sp_est.shape
    MSE = mean_squared_error(power_sp, power_sp_est)
    print(MSE)
    new_fn = os.path.basename(fn).replace("noised_", "")
    output_fn = f"../submit/{new_fn}"
    np.save(file=output_fn, arr=power_sp_est, allow_pickle=False, fix_imports=False)
    print(f"save {output_fn}.")

def save_to_zip():
    shutil.make_archive('submit-data', 'zip', '../submit')

def save_files(db, db_est, fn, power_sp, power_max, save_audio=False, plot=False):
    plot_db(db, db_est, fn)
    power_sp_est = np.power(10.0, 0.1 * db_est)
    power_sp_est_max = np.max(power_sp_est)
    power_sp_est *= (power_max / power_sp_est_max)
    calc_mse(power_sp, power_sp_est, fn)
    if save_audio:
        audio = librosa.feature.inverse.mel_to_audio(power_sp_est, sr=sr)
        write_wav(audio, fn)
    if plot:
        plot_3d(power_sp_est)

def bm3d_denoise(fn, save_audio):
    power_sp = np.load(fn)
    power_max = np.max(power_sp)
    db, db_abs_max = power_to_db(power_sp)
    z = np.atleast_3d(db)

    db_est = bm3d(z, np.sqrt(smoothing_factor), profile=profile)
    db_est = fix_denoised_db(db_denoised=db_est, db_abs_max=db_abs_max)
    save_files(db, db_est, fn, power_sp, power_max, save_audio=save_audio)

def main():
    for dir in "../audio ../submit ../fig".split():
        os.makedirs(dir, exist_ok=True)

    root_dir = "../dist-data/noised_tgt"
    for file in glob.glob(f"{root_dir}/*.npy"):
        bm3d_denoise(file, save_audio=False)

    with open("../submit/readme.txt", "w") as f:
        explanation = f"""
        1. Scaling mel-spectrogram using log function (10*np.log10(power_sp))
        2. Smooth log mel-spectrogram with Block mathing 3D
        3. Rescaling log mel-spectrogram (np.power(10.0, 0.1 * db_est))
        [bm3d parameters]
        {parameters}
        """
        f.write(explanation)

    save_to_zip()

def sample():
    imagename = 'cameraman256.png'

    # Load noise-free image
    y = np.array(Image.open(imagename)) / 255
    # Possible noise types to be generated 'gw', 'g1', 'g2', 'g3', 'g4', 'g1w', 'g2w', 'g3w', 'g4w'.
    noise_type = 'g3'
    noise_var = 0.02  # Noise variance
    seed = 0  # seed for pseudorandom noise realization

    # Generate noise with given PSD
    noise, psd, kernel = get_experiment_noise(noise_type, noise_var, seed, y.shape)
    z = np.atleast_3d(y) + np.atleast_3d(noise)

    # Call BM3D With the default settings.
    y_est = bm3d(z, psd)
    # To include refiltering:
    y_est = bm3d(z, psd, 'refilter')

    # For other settings, use BM3DProfile.
    profile = BM3DProfile(); # equivalent to profile = BM3DProfile('np');
    profile.gamma = 6;  # redefine value of gamma parameter
    y_est = bm3d(z, psd, profile);

    # Note: For white noise, you may instead of the PSD also pass a standard deviation
    y_est = bm3d(z, np.sqrt(noise_var));

    psnr = get_psnr(y, y_est)
    print("PSNR:", psnr)

    psnr_cropped = get_cropped_psnr(y, y_est, [16, 16])
    print("PSNR cropped:", psnr_cropped)

    y_est = np.minimum(np.maximum(y_est, 0), 1)
    z_rang = np.minimum(np.maximum(z, 0), 1)
    plt.title("y, z, y_est")
    plt.imshow(np.concatenate((y, np.squeeze(z_rang), y_est), axis=1), cmap='gray')
    plt.show()



if __name__ == '__main__':
    main()