from utilities.vis import plot_json

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import json

# Plotting Settings
plt.rc('axes', titlesize=25)
plt.rc('axes', labelsize=20)

plt.rcParams['axes.grid'] = True
plt.rcParams['axes.axisbelow'] = True

matplotlib.rc('xtick', labelsize=24)
matplotlib.rc('ytick', labelsize=24)
####################

file_names = ["Evaluation_COMPAS.json","Evaluation_ADULT.json"]

fairness_measures = ["Statistical Parity", "Equal Opportunity"]

row_length = []

# First extract relevant metadata
for name in file_names:
    f = open(".././results/" + name)
    data = json.load(f)
    f.close()

    models = [key for key in data.keys() if "Parameter" not in key]

    model_data = data[models[0]]

    datasets = [key for key in model_data.keys() if key != "ModelParameters" and not "Parameters" in key]

    row_length.append(len(datasets))


fig, axs = plt.subplots(np.sum(row_length), len(fairness_measures), figsize=(14, 6))

if len(np.shape(axs)) == 1:
    axs = axs[np.newaxis, :]

offset = 0
for i,name in enumerate(file_names):
    f = open(".././results/" + name)
    data = json.load(f)
    f.close()

    add_label = len(file_names)-1 == i

    plot_json(data,axs[offset:offset+row_length[i],:],fairness_measures,add_xlabel=add_label)

    offset = offset + row_length[i]

plt.tight_layout()
plt.show()