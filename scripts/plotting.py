
import os
import matplotlib.pyplot as plt
import csv
from minionsai.experiment_tooling import get_experiments_directory
import numpy as np

run_dir = get_experiments_directory()
runs = [
    ("parallel_v1", os.path.join(run_dir, "parallel_v1", "metrics.csv")),
    ("conv_big", os.path.join(run_dir, "conv_big", "metrics.csv")),
]

x_axis = "rollouts/games"
metrics = ["loss/epoch_0/batch_00", "eval_winrate/GenDiscAgent", "rollouts/game/0/kills"]

num = len(metrics)
fig, ax = plt.subplots(1, num, figsize=(num * 4, 4))
for label, csv_path in runs:

    # hack for a old runs. TODO: Remove this once we don't care about that run anymore
    if label == "conv_big":
        x_axis = "rollout_games"

    # Read data from the csv
    data = {metric: [] for metric in metrics}
    x = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row[x_axis] == "":
                continue
            
            for metric in metrics:
                if row[metric] == "":
                    data[metric].append(None)
                else:
                    data[metric].append(float(row[metric]))
            x.append(float(row[x_axis]))
    # Plot the data
    for i, metric in enumerate(metrics):
        valid = [i for i, v in enumerate(data[metric]) if v is not None]
        filtered_data = [data[metric][i] for i in valid]
        filtered_x = [x[i] for i in valid]
        # Plot with small dots
        ax[i].scatter(filtered_x, filtered_data, alpha=0.5, s=1)

        # Smooth using a moving average
        if len(filtered_data) > 20:
            smooth_size = 11
            smoothed = np.convolve(filtered_data, np.ones((smooth_size,)) / smooth_size, mode="valid")
            filtered_x = filtered_x[(smooth_size - 1)//2:-(smooth_size - 1)//2]
        else:
            smoothed = filtered_data
        ax[i].plot(filtered_x, smoothed, label=label)

        ax[i].set_xlabel(x_axis)
        # if x_axis == "rollout_games":
        ax[i].set_xlim(0, 51200)  # 256 * 200 = iteration 200 at 256 episodes_per_iteration
        ax[i].set_ylabel(metric)
        ax[i].legend()
plt.show()
