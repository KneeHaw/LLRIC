import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def parse_file(root: str, model_name: str):
    file_path = Path(root) / Path(model_name + ".log")
    with open(file_path, 'r') as f:
        _data = f.readlines()

    def line_to_datapoint(line: str):
        line = line[line.find("Loss: "):]
        pieces = line.replace(' ', '').replace('\n', '').split('|')
        pieces = [piece.split(':') for piece in pieces]
        pieces = [piece for piece in pieces if len(piece) > 1]

        datapoint = {}
        for pair in pieces:
            [_key, _value] = pair
            datapoint[_key.replace('loss', 'Loss')] = float(_value)

        return datapoint

    train_data = []
    for line in reversed(_data):
        if line.find("/20") > -1:
            datapoint = line_to_datapoint(line)
            if datapoint:
                train_data.insert(0, datapoint)

    test_data = []
    for line in reversed(_data):
        if line.find("Test epoch") > -1:
            datapoint = line_to_datapoint(line)
            if datapoint:
                test_data.insert(0, datapoint)

    return train_data[::20], test_data


def plot_losses(data: list[dict], x_label: str, title: str, steps_per_epoch: int = 1):
    """
    Plot all loss types overlaid on a single figure with different colors.

    Parameters
    ----------
    data             : list of dicts, e.g. [{'Loss': 18.9, 'MSELoss': 0.027, ...}, ...]
    x_label          : label for the x-axis, e.g. "Epoch"
    title            : overall figure title, e.g. "Train Losses" or "Test Losses"
    steps_per_epoch  : number of data points per epoch (default 1).
                       Set to 20 for train data so the x-axis is expressed in
                       fractional epochs with integer-epoch tick marks.
    """
    if not data:
        print(f"No data to plot for '{title}'.")
        return

    # Collect every loss key that appears across all datapoints
    loss_keys = list(dict.fromkeys(k for dp in data for k in dp))

    # X values expressed in epochs (fractional when steps_per_epoch > 1)
    x = np.arange(max(len(data), 100)) / steps_per_epoch

    fig, base_ax = plt.subplots(figsize=(15, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    axes = []
    for i, key in enumerate(loss_keys):
        if i == 0:
            ax = base_ax
        else:
            ax = base_ax.twinx()
            # Offset each extra y-axis outward so they don't overlap
            ax.spines['right'].set_position(('outward', 60 * (i - 1)))

        color = colors[i % len(colors)]
        values = [dp.get(key, float('nan')) for dp in data]
        values.extend([values[-1]] * (100 - len(values)))
        ax.plot(x, values, linewidth=1.5, color=color, label=key)
        if key != 'AuxLoss':
            ax.set_yscale('log')
        ax.set_ylabel(key, color=color, fontsize=10)
        ax.tick_params(axis='y', colors=color)
        ax.spines['right'].set_edgecolor(color)
        if i == 0:
            ax.spines['left'].set_edgecolor(color)
            ax.yaxis.label.set_color(color)
        axes.append(ax)

    # Tick at every integer epoch
    max_epoch = int(np.floor(x[-1]))
    base_ax.set_xticks(np.arange(0, 100 + 1, step=10))
    base_ax.set_xlabel(x_label, fontsize=11)
    base_ax.set_xlim(0, x[-1])
    base_ax.grid(True, linestyle='--', alpha=0.5)

    # Combine legends from all axes into one
    lines, labels = [], []
    for ax in axes:
        l, lb = ax.get_legend_handles_labels()
        lines.extend(l)
        labels.extend(lb)
    base_ax.legend(lines, labels, fontsize=10, loc='upper right')

    # Make room on the right for the extra y-axes
    n_right_axes = len(loss_keys) - 1
    fig.subplots_adjust(right=max(0.5, 1.0 - 0.1 * n_right_axes))
    return fig


def main():
    root = "./checkpoints/tinylic/3/"
    train_data, test_data = parse_file(root, "partial_llric_test1")

    print(train_data[-1], test_data[-1])
    plot = True
    if plot:
        train_fig = plot_losses(
            train_data,
            x_label="Epoch",
            title="Train Losses",
            steps_per_epoch=20,
        )

        test_fig = plot_losses(
            test_data,
            x_label="Epoch",
            title="Test Losses",
        )

        if train_fig:
            train_fig.savefig("partial_train_losses.png", dpi=150, bbox_inches='tight')

        if test_fig:
            test_fig.savefig("partial_test_losses.png", dpi=150, bbox_inches='tight')

if __name__ == "__main__":
    main()