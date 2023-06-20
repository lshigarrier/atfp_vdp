import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm


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


def main():
    colors    = plt.cm.jet
    colorlist = [colors(i) for i in range(colors.N)]
    print(colorlist)


if __name__ == '__main__':
    main()
