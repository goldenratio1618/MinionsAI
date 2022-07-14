
import os
import matplotlib.pyplot as plt
import csv
from minionsai.experiment_tooling import get_experiments_directory
import numpy as np

run_dir = get_experiments_directory()
runs = [
    ("disc_from_scratch", os.path.join(run_dir, "disc_from_scratch", "metrics.csv")),
    ("disc_32epi_3e5lr", os.path.join(run_dir, "disc_32epi_3e5lr", "metrics.csv")),

    # # ("test", os.path.join(run_dir, "test", "metrics.csv")),
    # # ("dumbrandom_nostop_t03_eps04", os.path.join(run_dir, "dumbrandom_nostop_t03_eps04", "metrics.csv")),
    # ("gen_conveps400_2", os.path.join(run_dir, "gen_conveps400_2", "metrics.csv")),
    # ("gen_conveps396_3", os.path.join(run_dir, "gen_conveps396_3", "metrics.csv")),
    # ("gen_conveps396_5", os.path.join(run_dir, "gen_conveps396_5", "metrics.csv")),
    # ("gen_convbig396", os.path.join(run_dir, "gen_convbig396", "metrics.csv")),

    # ("conveps_repro_0704", os.path.join(run_dir, "conveps_repro_0704", "metrics.csv")),
    # ("conv_big", os.path.join(run_dir, "conv_big", "metrics.csv")),
    # ("cycle_bot", os.path.join(run_dir, "cycle_bot", "metrics.csv")),
    # ("cycle_bot_2", os.path.join(run_dir, "cycle_bot_2", "metrics.csv")),
    # ("cycle_bot_3", os.path.join(run_dir, "cycle_bot_3", "metrics.csv")),
]

x_axis = "rollouts/games"
# x_axis = "iteration"
metrics = [
    ["gen/loss/epoch_0/batch_0000", "gen/loss/epoch_0/batch_000", "disc/loss/epoch_0/batch_000"], 
    ["eval_winrate/dfarhi_0613_conveps_256rolls_iter400_adapt"],
    ["timing/iteration"]
]

num = len(metrics)
fig, ax = plt.subplots(1, num, figsize=(num * 6, 5))

flat_metrics = [metric  for metrics_list in metrics for metric in metrics_list]
for label, csv_path in runs:

    # Read data from the csv
    data = {metric: [] for metric in flat_metrics}
    x = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row[x_axis] == "":
                continue
            
            for metric in flat_metrics:
                if metric not in row or row[metric] == "":
                    data[metric].append(None)
                else:
                    data[metric].append(float(row[metric]))
            x.append(float(row[x_axis]))
    # Plot the data
    for i, metrics_this_plot in enumerate(metrics):
        for metric in metrics_this_plot:
            valid = [i for i, v in enumerate(data[metric]) if v is not None]
            filtered_data = [data[metric][i] for i in valid]
            filtered_x = [x[i] for i in valid]
            # Plot with small dots
            if len(filtered_data) > 0:
                ax[i].scatter(filtered_x, filtered_data, alpha=0.5, s=1)

            # Smooth using a moving average
            if len(filtered_data) > 50:
                smooth_size = 11
                smoothed = np.convolve(filtered_data, np.ones((smooth_size,)) / smooth_size, mode="valid")
                filtered_x = filtered_x[(smooth_size - 1)//2:-(smooth_size - 1)//2]
            else:
                smoothed = filtered_data
            style = "-"
            if len(smoothed) > 0:
                ax[i].plot(filtered_x, smoothed, label=label, linestyle=style)

        ax[i].set_xlabel(x_axis)
        # if x_axis == "rollout_games":
        # ax[i].set_xlim(0, 400)
        ax[i].set_ylabel(metric)
        ax[i].legend()
plt.show()
