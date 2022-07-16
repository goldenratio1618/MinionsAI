
import os
import matplotlib.pyplot as plt
import csv
from minionsai.experiment_tooling import get_experiments_directory
import numpy as np

run_dir = get_experiments_directory()
runs = [
    ("dual_1", os.path.join(run_dir, "dual_1", "metrics.csv")),

    # ("disc_from_scratch", os.path.join(run_dir, "disc_from_scratch", "metrics.csv")),
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

# x_axis = "rollouts/games"
x_axis = "iteration"
metrics = [
    # ["gen/loss/epoch_0/batch_00", "gen/loss/epoch_0/batch_000"],
    ["disc/loss/epoch_0/batch_000"], 
    ["eval_winrate/dfarhi_0613_conveps_256rolls_iter400_adapt"],
    ["generator_strength"],
    ["timing/iteration"]
]

num = len(metrics)
fig, ax = plt.subplots(1, num, figsize=(num * 6, 5))

flat_metrics = [metric  for metrics_list in metrics for metric in metrics_list]
if any("generator_strength" in x for x in metrics):
    flat_metrics.extend(["rollouts/game/generators/0/have_best_action", "rollouts/game/generators/1/have_best_action"])

def float_lookup(key, dict):
    if key not in dict or dict[key] == "":
        return None
    return float(dict[key])

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
                if metric == "generator_strength":
                    gen_str = float_lookup("rollouts/game/generators/0/have_best_action", row)
                    random_str = float_lookup("rollouts/game/generators/1/have_best_action", row)
                    if gen_str is not None:
                        data[metric].append(gen_str - random_str)
                else:
                    data[metric].append(float_lookup(metric, row))
                
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
            if len(filtered_data) > 100:
                smooth_size = 61
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
ax[2].grid(True)
plt.show()
