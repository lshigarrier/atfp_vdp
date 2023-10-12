import numpy as np
import scipy
import pickle
import io
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
from matplotlib.widgets import Slider
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, Normalize
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from utils import load_yaml


def moving_average(array, window_size):
    i = 0
    moving_averages = []
    # array = [*array, *[array[-1] for _ in range(window_size-1)]]
    # array = np.array(array)
    # while i < len(array) - window_size + 1:
    n = len(array)
    while i < n:
        if i + window_size > n:
            array.append(moving_averages[-1])
        # mean = np.exp(np.log(array[i: i + window_size]).mean())
        mean = np.array(array[i: i + window_size]).mean()
        # mean = array[i: i + min(window_size, len(array)-i)].mean()
        moving_averages.append(mean)
        i += 1
    return moving_averages


def plot_hist(data, bins=50, title=None, xlabel=None, ylabel=None):
    fig, ax = plt.subplots(figsize=(24, 18))
    ax.hist(data, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig


def plot_boxplot(data, title='Predicted variance'):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.boxplot(data)
    return fig


def plot_curves(data, legend, title, xlabel, ylabel, xlim=(None, None), ylim=(None, None), stop=None, val=False):
    fig, ax = plt.subplots(figsize=(24, 18))
    add_legend = False
    for i in range(len(data)):
        if legend[i] is None:
            if stop is not None:
                if i==1 and val:
                    ax.plot(np.linspace(1, stop, num=len(data[i]), endpoint=True), data[i])
                else:
                    ax.plot(np.linspace(0, stop, num=len(data[i]), endpoint=False), data[i])
            else:
                ax.plot(range(1, len(data[i])+1), data[i])
        else:
            add_legend = True
            if stop is not None:
                if i==1 and val:
                    ax.plot(np.linspace(1, stop, num=len(data[i]), endpoint=True), data[i], label=legend[i])
                else:
                    ax.plot(np.linspace(0, stop, num=len(data[i]), endpoint=False), data[i], label=legend[i])
            else:
                ax.plot(range(1, len(data[i])+1), data[i], label=legend[i])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if add_legend:
        ax.legend()
    return fig


def plot_spot(preds, truth, varis=None):
    fig, ax = plt.subplots(figsize=(24, 18))
    ax.plot(range(len(truth)), truth, label='Truth')
    ax.plot(range(len(preds)), preds, label='Prediction')
    ax.set_xlabel('Time')
    ax.set_ylabel('Class')
    ax.legend()
    if varis is not None:
        ax2 = ax.twinx()
        ax2.plot(range(len(varis)), varis, label='Variance', c='red')
        ax2.set_ylabel('Variance')
        ax2.legend()
    return fig


def plot_one_img(tsr):
    """
    :param tsr: tensor with shape time x width x height
    :return: fig
    """
    nb_lon = tsr.shape[0]
    nb_lat = tsr.shape[1]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect('equal')
    _ = ax.imshow(tsr, cmap='Greys', interpolation='none', aspect="auto")
    ax.set_xlim(0, nb_lon)
    ax.set_ylim(0, nb_lat)
    return fig


def plot_pred(preds, truth, no_zero=False, nb_classes=5, t_init=70):
    """
    Plot the predicted congestion and the true congestion
    preds: nb_times x nb_lon x nb_lat
    truth: nb_times x nb_lon x nb_lat
    """
    nb_times = preds.shape[0]
    nb_lon   = preds.shape[1]
    nb_lat   = preds.shape[2]
    mask     = truth[0].ne(-1).int()
    if no_zero:
        mask = mask*truth[t_init].ne(0).int()

    # Create colormap
    colors    = plt.cm.jet
    colorlist = [colors(i) for i in range(colors.N)]
    colorlist[0] = (1., 1., 1., 1.)
    cmap      = LinearSegmentedColormap.from_list('cmap', colorlist, colors.N)
    bounds    = np.linspace(0, nb_classes, nb_classes+1)
    norm      = BoundaryNorm(bounds, colors.N)

    fig, ax = plt.subplots(1, 3, figsize=(12, 5), width_ratios=[1, 10, 10])
    ax[1].set_aspect('equal')
    ax[2].set_aspect('equal')
    plt.subplots_adjust(bottom=0.2)
    im_true = ax[1].imshow(mask*truth[t_init, ...], cmap=cmap, norm=norm, aspect="auto")
    im_pred = ax[2].imshow(mask*preds[t_init, ...], cmap=cmap, norm=norm, aspect="auto")
    ax[1].set_xlim(0, nb_lon)
    ax[1].set_ylim(0, nb_lat)
    ax[2].set_xlim(0, nb_lon)
    ax[2].set_ylim(0, nb_lat)
    ax[1].set_title("True")
    ax[2].set_title("Predicted")

    mpl.colorbar.ColorbarBase(ax[0], cmap=cmap, norm=norm,
                              spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i')
    ax[0].yaxis.set_ticks_position('left')

    axe_slider = fig.add_axes([0.1, 0.01, 0.8, 0.05])
    slider     = Slider(axe_slider, 'slider', 0, nb_times - 1, valinit=t_init, valstep=1)

    def update_eval(val):
        time = int(slider.val)
        new_mask = truth[0].ne(-1).int()
        if no_zero:
            new_mask = new_mask*truth[time].ne(0).int()
        im_true.set_data(new_mask*truth[time, ...])
        im_pred.set_data(new_mask*preds[time, ...])
        plt.draw()

    slider.on_changed(update_eval)
    return fig, slider


def plot_pred_vdp(preds, truth, varis, no_zero=False, nb_classes=5, var_range=None, t_init=70):
    """
    Plot the predicted congestion, the true congestion, and the variance
    preds: nb_times x nb_lon x nb_lat
    truth: nb_times x nb_lon x nb_lat
    """
    nb_times = preds.shape[0]
    nb_lon   = preds.shape[1]
    nb_lat   = preds.shape[2]
    mask     = truth[0].ne(-1).int()
    if no_zero:
        mask = mask*truth[t_init].ne(0).int()

    # Describe variances
    print(f'Statistics of predicted variances: {scipy.stats.describe(varis.flatten())}')
    print(f'Median of predicted variances: {varis.flatten().median()}')

    # Create colormap for classes
    colors    = plt.cm.jet
    colorlist = [colors(i) for i in range(colors.N)]
    colorlist[0] = (1., 1., 1., 1.)
    cmap      = LinearSegmentedColormap.from_list('cmap', colorlist, colors.N)
    bounds    = [i for i in range(nb_classes + 1)]
    norm      = BoundaryNorm(bounds, colors.N)
    if var_range is None:
        var_min = varis.min()
        var_max = varis.max()
    else:
        var_min = var_range[0]
        var_max = var_range[1]
    norm_var  = Normalize(vmin=var_min, vmax=var_max)

    fig, ax = plt.subplots(1, 6, figsize=(20, 4), width_ratios=[1, 10, 10, 10, 10, 1])
    ax[1].set_aspect('equal')
    ax[2].set_aspect('equal')
    ax[3].set_aspect('equal')
    ax[4].set_aspect('equal')
    im_true = ax[1].imshow(mask*truth[t_init, ...], cmap=cmap, norm=norm, aspect="auto")
    im_pred = ax[2].imshow(mask*preds[t_init, ...], cmap=cmap, norm=norm, aspect="auto")
    im_var  = ax[3].imshow(mask*varis[t_init, ...], cmap='viridis', norm=norm_var, aspect="auto")
    im_err  = ax[4].imshow(mask*torch.abs(preds[t_init, ...] - truth[t_init, ...]), cmap=cmap, norm=norm, aspect="auto")
    ax[1].set_xlim(0, nb_lon)
    ax[1].set_ylim(0, nb_lat)
    ax[2].set_xlim(0, nb_lon)
    ax[2].set_ylim(0, nb_lat)
    ax[3].set_xlim(0, nb_lon)
    ax[3].set_ylim(0, nb_lat)
    ax[4].set_xlim(0, nb_lon)
    ax[4].set_ylim(0, nb_lat)
    ax[1].set_title("True")
    ax[2].set_title("Predicted")
    ax[3].set_title("Variance")
    ax[4].set_title("Error")

    plt.subplots_adjust(bottom=0.2)
    mpl.colorbar.ColorbarBase(ax[0], cmap=cmap, norm=norm,
                              spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i')
    ax[0].yaxis.set_ticks_position('left')
    mpl.colorbar.ColorbarBase(ax[5], cmap='viridis', norm=norm_var,
                              ticks=np.linspace(var_min, var_max, nb_classes+1))
    axe_slider = fig.add_axes([0.1, 0.01, 0.8, 0.05])
    slider     = Slider(axe_slider, 'slider', 0, nb_times - 1, valinit=t_init, valstep=1)

    def update_eval(val):
        time = int(slider.val)
        new_mask = truth[0].ne(-1).int()
        if no_zero:
            new_mask = new_mask*truth[time].ne(0).int()
        im_true.set_data(new_mask*truth[time, ...])
        im_pred.set_data(new_mask*preds[time, ...])
        im_var.set_data(new_mask*varis[time, ...])
        im_err.set_data(new_mask*torch.abs(preds[time, ...] - truth[time, ...]))
        plt.draw()

    slider.on_changed(update_eval)
    return fig, slider


def get_ylim(array_list):
    ymin = array_list[0].mean()
    ymax = array_list[0].mean()
    for i in range (1, len(array_list)):
        ymin = min(ymin, array_list[i].mean())
        ymax = max(ymax, array_list[i].mean())
    rang = 2*(ymax - ymin)
    ymax = ymax + rang
    ymin = ymin - rang
    if np.isnan(ymax):
        ymax = None
    if np.isnan(ymin):
        ymin = None
    return ymin, ymax


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


def main():
    param = load_yaml('plots')
    if param['no_zero']:
        nb_class = param['nb_classes'] + 1
    else:
        nb_class = param['nb_classes']

    # Load data
    preds, truth = [], []
    if param['dataset'] == 'pirats':
        preds = torch.load(f'models/{param["name"]}/preds.pickle')
        truth = torch.load(f'models/{param["name"]}/truth.pickle')

    window = param['window']
    with open(f'models/{param["name"]}/loss.pickle', 'rb') as f:
        loss_full = pickle.load(f)
        loss_full = np.array(moving_average(loss_full, window))
    with open(f'models/{param["name"]}/loss_val.pickle', 'rb') as f:
        loss_val = pickle.load(f)
        loss_val = np.array(loss_val)
    with open(f'models/{param["name"]}/nll.pickle', 'rb') as f:
        nll_full = pickle.load(f)
        nll_full = np.array(moving_average(nll_full, window))
    with open(f'models/{param["name"]}/nll_val.pickle', 'rb') as f:
        nll_val = pickle.load(f)
        nll_val = np.array(nll_val)
    with open(f'models/{param["name"]}/kl.pickle', 'rb') as f:
        kl_full = pickle.load(f)
        kl_full = np.array(moving_average(kl_full, window))
    # with open(f'models/{param["name"]}/cutoff.pickle', 'rb') as f:
    #     b_full = pickle.load(f)
    with open(f'models/{param["name"]}/class.pickle', 'rb') as f:
        class_full = pickle.load(f)
        # class_full = CPU_Unpickler(f).load()
    for i in range(len(class_full)):
        # class_full[i] = [_.item() for _ in class_full[i]]
        class_full[i] = np.array(moving_average(class_full[i], window))

    # Plots
    if param['dataset'] == 'pirats':
        truth_flat = truth.flatten()
        preds_flat = preds.flatten()
        mask = truth_flat.ne(-1)
        if param['no_zero']:
            mask = mask*truth_flat.ne(0)
        truth_flat = truth_flat[mask]
        preds_flat = preds_flat[mask]
        cm = confusion_matrix(truth_flat, preds_flat)
        ConfusionMatrixDisplay(cm).plot()
    figs = []

    if param['vdp']:
        varis = torch.load(f'models/{param["name"]}/varis.pickle')
        if param['dataset'] == 'pirats':
            if param['predict_spot']:
                figs.append(plot_spot(preds, truth, varis))
            else:
                figs.append(plot_pred_vdp(preds, truth, varis, param['no_zero'], nb_class, param['var_range']))
            figs.append(plot_boxplot(varis.flatten()))
        var_correct = torch.load(f'models/{param["name"]}/var_corr.pickle')
        var_incorr  = torch.load(f'models/{param["name"]}/var_incorr.pickle')
        if param['plot_hist']:
            figs.append(plot_hist(var_correct, bins=50, title='Correctly classified',
                                  xlabel='Variance', ylabel='Numbers'))
            figs.append(plot_hist(var_incorr, bins=50, title='Incorrectly classified',
                                  xlabel='Variance', ylabel='Numbers'))
        else:
            figs.append(plot_boxplot(var_correct, 'Correctly classified'))
            figs.append(plot_boxplot(var_incorr, 'Incorrectly classified'))

    else:
        if param['dataset'] == 'pirats':
            if param['predict_spot']:
                figs.append(plot_spot(preds, truth))
            else:
                figs.append(plot_pred(preds, truth, param['no_zero'], nb_class))

    print(f'KL factor: {param["kl_factor"]}')
    figs.append(plot_curves([loss_full, loss_val], ['Training', 'Validation'],
                            'Full loss', 'Epoch', 'Full loss',
                            ylim=get_ylim([loss_full, loss_val]),
                            stop=param['epochs'], val=True))
    figs.append(plot_curves([nll_full, nll_val], ['Training', 'Validation'],
                            'NLL', 'Epoch', 'Loss', ylim= get_ylim([nll_full, nll_val]),
                            stop=param['epochs'], val=True))
    figs.append(plot_curves([param['kl_factor']*kl_full], [None], 'factor*KL', 'Epoch', 'Loss', stop=param['epochs']))
    figs.append(plot_curves([nll_full, param['kl_factor']*kl_full],
                            ['nll', 'factor*kl'], 'NLL and factor*KL', 'Epoch', 'Loss',
                            ylim=get_ylim([nll_full, param['kl_factor']*kl_full]), stop=param['epochs']))
    # figs.append(plot_curves(b_full, ['b1', 'b2'],
    #                         'Cutoff parameters during training', 'Epoch', 'Cutoff parameters', stop=param['epochs']))
    if param['no_zero']:
        class_legend = [f'Class {i+1}' for i in range(len(class_full))]
    else:
        class_legend = [f'Class {i}' for i in range(len(class_full))]
    figs.append(plot_curves(class_full, class_legend, 'NLL for each class', 'Epoch', 'Loss',
                            ylim=get_ylim(class_full), stop=param['epochs']))
    plt.show()


if __name__ == '__main__':
    main()
