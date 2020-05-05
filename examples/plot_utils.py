import os
import numpy as np
import matplotlib.pyplot as plt

class show_close:
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_val, exc_tb):
        plt.show()
        plt.close()


def imshow(signal):
    with show_close as sc:
        plt.imshow(signal)

def histogram(signal):
    with show_close as sc:
        plt.hist(signal)

def plot_qualtile(signal):
    with show_close as sc:
        fig, axes = plt.subplots(1, 2)
        axes[0].boxplot(signal.flatten())
        axes[1].violinplot(signal.flatten())

def plot_db(db, db_denoised, fn):
    fig, axes = plt.subplots(2, 1)
    for ax, im in zip(axes, [db, db_denoised]):
        ax.imshow(im)
    output_fn = os.path.basename(fn).replace(".npy", ".png")
    output_fig = f"../fig/{output_fn}"
    plt.savefig(output_fig)
    plt.close()

def plot_3d(power_sp, power_sp_est, fn):
    assert power_sp.shape == power_sp_est.shape
    y, x = power_sp.shape
    xx, yy = np.meshgrid(np.arange(x), np.arange(y))

    fig = plt.figure()
    for i, (power, name) in enumerate(zip([power_sp, power_sp_est], ["power_sp", "power_sp_est"])):
        ax = fig.add_subplot(1, 2, i+1, projection='3d')
        ax.set_xlabel("x")
        ax.set_ylabel("freq")
        ax.set_zlabel("magnitude")
        ax.set_title(name)
        ax.plot_surface(xx, yy, power, cmap='viridis')

    out_dir = "../power_spectrum"
    new_fn = os.path.join(out_dir,
                          os.path.basename(fn).replace(".npy", ".png"))
    plt.savefig(new_fn)
    plt.close()