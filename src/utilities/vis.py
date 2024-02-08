import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def visualize_result(ax,model):
    labels = model.get_prototype_labels()
    positions = model.get_prototype_positions()
    ax.scatter(positions[0,:], positions[1,:], c=labels, marker="^",s=200,  cmap="tab10", edgecolors='black', vmin=min(labels),vmax=max(labels))


def visualize_result_and_prediction(ax,X,protected,y_prediction,model,title):
    markers = ["o", "s"]
    colors = ["skyblue", "salmon", "yellowgreen", "plum"]

    labels = model.get_prototype_labels()
    positions = model.get_prototype_positions()

    for i, unique in enumerate(np.unique(protected)):
        idx = np.where(protected == i)
        color = [colors[int(value)] for value in np.array(y_prediction)[idx]]
        ax.scatter(X[idx, 0], X[idx, 1], edgecolors="black", c=color, s=150, marker=markers[i])

    for i,label in enumerate(labels):
        ax.scatter(positions[0,i], positions[1,i], c=colors[label], marker="^",s=220, edgecolors='black')

    ax.set_title(title,fontsize=24)
    ax.axis("equal")
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))




def plot_json(data,axs,fairness_measures,add_xlabel=True):
    def set_linestyle(errorbars):
        for eb in errorbars[-1]:
            eb.set_linestyle(lstyle[models.index(model)])

    models = [key for key in data.keys() if "Parameter" not in key]

    markers = ['s', '.', '*']
    colors = ['steelblue', 'mediumorchid', 'lightcoral']
    lstyle = ['solid', 'dashed', 'dotted']
    lw = 2

    ####

    for model in models:
        print(model)
        model_data = data[model]

        # Keys are datasets except "ModelParameters"
        datasets = [key for key in model_data.keys() if key != "ModelParameters" and not "Parameters" in key]

        # Only one dataset
        # datasets = [data for data in datasets if data == "COMPAS"]  #ADULT (sex)


        for i, dataset in enumerate(datasets):
            parameters = model_data["ModelParameters"]

            print(dataset)
            model_dataset_data = model_data[dataset]


            st_parity_means = []
            accuracies_means = []
            eq_odd_means = []

            st_parity_stds = []
            accuracies_stds = []
            eq_odd_stds = []

            for j, parameter in enumerate(parameters):
                evaluation_result = model_dataset_data[str(tuple(parameter))]

                # legacy
                # evaluation_result = model_dataset_data[str(parameter)]

                accuracies = evaluation_result["Accuracy"]
                st_parities = evaluation_result["StatisticalParity"]
                eq_odds = evaluation_result["EqualOpportunity"]

                st_parity_means.append(np.mean(st_parities))
                accuracies_means.append(np.mean(accuracies))
                eq_odd_means.append(np.mean(eq_odds))

                st_parity_stds.append(np.std(st_parities))
                accuracies_stds.append(np.std(accuracies))
                eq_odd_stds.append(np.std(eq_odds))

            if model == "FGLVQ":
                legend_name = "FairGLVQ"
                ms = 8
            elif model == "Dummy":
                legend_name = "Const."
                ms = 15
            else:
                legend_name = model
                ms = 15

            ebs = axs[i, 0].errorbar(st_parity_means, accuracies_means, xerr=st_parity_stds, yerr=accuracies_stds,
                                     capsize=6,
                                     c=colors[models.index(model)], ms=ms, marker=markers[models.index(model)],
                                     label=legend_name, linestyle=lstyle[models.index(model)], linewidth=lw, capthick=2,
                                     markeredgecolor="black")

            set_linestyle(ebs)


            ebs = axs[i, 1].errorbar(eq_odd_means, accuracies_means, xerr=eq_odd_stds,
                                     yerr=accuracies_stds, capsize=6,
                                     c=colors[models.index(model)], ms=ms, marker=markers[models.index(model)],
                                     label=legend_name, linestyle=lstyle[models.index(model)], linewidth=lw, capthick=2,
                                     markeredgecolor="black")

            set_linestyle(ebs)

    for ax, row in zip(axs[:, 0], datasets):
        if 'ADULT' in row:
            lp = 8  # 18
            row = "Adult"

        elif 'COMPAS' in row:
            lp = 8
        else:
            lp = 10

        plt.text(0.01, 0.97, row,
                 horizontalalignment='left',
                 verticalalignment='top',
                 transform=ax.transAxes,
                 in_layout=False,
                 bbox=dict(edgecolor='black', facecolor="white", fill=True),
                 fontsize=18)


        ax.set_ylabel("Accuracy", rotation=90, labelpad=lp, fontsize=24)  # , fontsize=30

    for ax, row in zip(axs[:, 1], datasets):
        if 'ADULT' in row:
            lp = 8  # 18
            row = "Adult"

        elif 'COMPAS' in row:
            lp = 8
        else:
            lp = 10
        plt.text(0.01, 0.97, row,
                 horizontalalignment='left',
                 verticalalignment='top',
                 transform=ax.transAxes,
                 in_layout=False,
                 bbox=dict(edgecolor='black', facecolor="white", fill=True),
                 fontsize=18)

    for ax, col in zip(axs[-1,:], fairness_measures):
        if add_xlabel == True:
            ax.set_xlabel(col, fontsize=24)


    plt.legend(loc='lower right', fontsize=17)
