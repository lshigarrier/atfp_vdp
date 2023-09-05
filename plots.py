import numpy as np
import scipy
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, Normalize
from utils import load_yaml


def plot_spot(preds, truth):
    fig, ax = plt.subplots(figsize=(24, 18))
    ax.plot(range(len(truth)), truth, label='Truth')
    ax.plot(range(len(preds)), preds, label='Prediction')
    ax.set_xlabel('Time')
    ax.set_ylabel('Class')
    ax.legend()
    return fig


def plot_pred(preds, truth, nb_classes=5, t_init=70):
    """
    Plot the predicted congestion and the true congestion
    preds: nb_times x nb_lon x nb_lat
    truth: nb_times x nb_lon x nb_lat
    """
    nb_times = preds.shape[0]
    nb_lon   = preds.shape[1]
    nb_lat   = preds.shape[2]

    # Create colormap
    colors    = plt.cm.jet
    colorlist = [colors(i) for i in range(colors.N)]
    colorlist[0] = (1., 1., 1., 1.)
    cmap      = LinearSegmentedColormap.from_list('cmap', colorlist, colors.N)
    bounds    = np.linspace(0, nb_classes, nb_classes+1)
    norm      = BoundaryNorm(bounds, colors.N)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')
    plt.subplots_adjust(bottom=0.2)
    im_pred = ax[0].imshow(preds[t_init, ...], cmap=cmap, norm=norm, aspect="auto")
    im_true = ax[1].imshow(truth[t_init, ...], cmap=cmap, norm=norm, aspect="auto")
    ax[0].set_xlim(0, nb_lon)
    ax[0].set_ylim(0, nb_lat)
    ax[1].set_xlim(0, nb_lon)
    ax[1].set_ylim(0, nb_lat)
    ax[0].set_title("Predicted")
    ax[1].set_title("True")

    axe_bar = fig.add_axes([0.95, 0.1, 0.03, 0.8])
    mpl.colorbar.ColorbarBase(axe_bar, cmap=cmap, norm=norm,
                              spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i')

    axe_slider = fig.add_axes([0.1, 0.01, 0.8, 0.05])
    slider     = Slider(axe_slider, 'slider', 0, nb_times - 1, valinit=t_init, valstep=1)

    def update_eval(val):
        time = int(slider.val)
        im_pred.set_data(preds[time, ...])
        im_true.set_data(truth[time, ...])
        plt.draw()

    slider.on_changed(update_eval)
    return fig, slider


def plot_pred_vdp(preds, truth, varis, nb_classes=5, t_init=70):
    """
    Plot the predicted congestion, the true congestion, and the variance
    preds: nb_times x nb_lon x nb_lat
    truth: nb_times x nb_lon x nb_lat
    """
    nb_times = preds.shape[0]
    nb_lon   = preds.shape[1]
    nb_lat   = preds.shape[2]

    # Describe variances
    print(f'Statistics of predicted variances: {scipy.stats.describe(varis.flatten())}')

    # Create colormap for classes
    colors    = plt.cm.jet
    colorlist = [colors(i) for i in range(colors.N)]
    colorlist[0] = (1., 1., 1., 1.)
    cmap      = LinearSegmentedColormap.from_list('cmap', colorlist, colors.N)
    bounds    = [i for i in range(nb_classes + 1)]
    norm      = BoundaryNorm(bounds, colors.N)
    norm_var  = Normalize(vmin=0, vmax=max(varis.max(), 1))

    fig, ax = plt.subplots(1, 5, figsize=(17, 5), width_ratios=[1, 10, 10, 10, 1])
    ax[1].set_aspect('equal')
    ax[2].set_aspect('equal')
    ax[3].set_aspect('equal')
    im_pred = ax[1].imshow(preds[t_init, ...], cmap=cmap, norm=norm, aspect="auto")
    im_true = ax[2].imshow(truth[t_init, ...], cmap=cmap, norm=norm, aspect="auto")
    im_var  = ax[3].imshow(varis[t_init, ...], cmap='viridis', norm=norm_var, aspect="auto")
    ax[1].set_xlim(0, nb_lon)
    ax[1].set_ylim(0, nb_lat)
    ax[2].set_xlim(0, nb_lon)
    ax[2].set_ylim(0, nb_lat)
    ax[3].set_xlim(0, nb_lon)
    ax[3].set_ylim(0, nb_lat)
    ax[1].set_title("Predicted")
    ax[2].set_title("True")
    ax[3].set_title("Variance")

    plt.subplots_adjust(bottom=0.2)
    mpl.colorbar.ColorbarBase(ax[0], cmap=cmap, norm=norm,
                              spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i')
    ax[0].yaxis.set_ticks_position('left')
    mpl.colorbar.ColorbarBase(ax[4], cmap='viridis', norm=norm_var,
                              ticks=np.linspace(0, max(varis.max(), 1), nb_classes+1))
    axe_slider = fig.add_axes([0.1, 0.01, 0.8, 0.05])
    slider     = Slider(axe_slider, 'slider', 0, nb_times - 1, valinit=t_init, valstep=1)

    def update_eval(val):
        time = int(slider.val)
        im_pred.set_data(preds[time, ...])
        im_true.set_data(truth[time, ...])
        im_var.set_data(varis[time, ...])
        plt.draw()

    slider.on_changed(update_eval)
    return fig, slider


def main():
    param = load_yaml('test_ed')
    # with open(f'{param["fig_file"]}.fig.pickle', 'rb') as file:
    #     _ = pickle.load(file)
    preds = torch.load(f'{param["fig_file"]}preds.pickle')
    truth = torch.load(f'{param["fig_file"]}truth.pickle')
    if param['vdp']:
        varis = torch.load(f'{param["fig_file"]}varis.pickle')
        _     = plot_pred_vdp(preds, truth, varis, param['nb_classes'])
    else:
        _ = plot_pred(preds, truth, param['nb_classes'])
    plt.show()


if __name__ == '__main__':
    main()
