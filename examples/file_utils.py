import os
import shutil
import librosa
from sklearn.metrics import mean_squared_error
from plot_utils import *

sr = 22050

def save_to_zip():
    shutil.make_archive('submit-data', 'zip', '../submit')

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

def save_files(db, db_est, fn, power_sp, save_audio=False, plot=False):
    plot_db(db, db_est, fn)
    power_sp_est = np.power(10.0, db_est)
    calc_mse(power_sp, power_sp_est, fn)
    if save_audio:
        audio = librosa.feature.inverse.mel_to_audio(power_sp_est, sr=sr)
        write_wav(audio, fn)
    if plot:
        plot_3d(power_sp, power_sp_est, fn)