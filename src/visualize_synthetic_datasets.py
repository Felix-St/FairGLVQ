from sklearn.preprocessing import MinMaxScaler
from utilities import distributions
from utilities.INP import get_debiasing_projection, get_debiasing_projection_with_intercept
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

ms = 400

def plot_synthetic_distribution(data,ax,title,ms):
    markers = ["o", "s"]
    colors = ["skyblue", "salmon", "yellowgreen", "plum"]
    for i, unique_prot in enumerate(np.unique(data[:, 2])):
        idx = np.where(data[:, 2] == unique_prot)
        data_filtered = np.reshape((data[idx, :]), (-1, 4))
        for j, unique_lab in enumerate(np.unique(data_filtered[:,3])):
            idx_second = np.where(data_filtered[:, 3] == unique_lab)

            color = [colors[int(value)] for value in data_filtered[idx_second, 3][0]]

            if unique_lab == 0:
                ax.scatter(data_filtered[idx_second, 0], data_filtered[idx_second, 1], edgecolors="black", c=color, s=ms, marker=markers[i],hatch='//')
            else:
                ax.scatter(data_filtered[idx_second, 0], data_filtered[idx_second, 1], edgecolors="black",c=color, s=ms, marker=markers[i])

    ax.axis('equal')
    if title != None:
        ax.set_title(title,fontsize=26)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])


matplotlib.rc('xtick', labelsize=40)
matplotlib.rc('ytick', labelsize=40)

plt.rc('axes', titlesize=25)
plt.rc('axes', labelsize=20)

XOR = distributions.create_synthetic_dataset_XOR(100,seed=20)
scaler = MinMaxScaler()
XOR_norm = scaler.fit_transform(XOR)

LOCAL = distributions.create_synthetic_dataset_Local(100,2,seed=20)
scaler = MinMaxScaler()
LOCAL_norm = scaler.fit_transform(LOCAL)

fig, axs = plt.subplots(1, 2, figsize=(13, 6))
plot_synthetic_distribution(LOCAL_norm,axs[0],None,ms=ms)
plot_synthetic_distribution(XOR_norm,axs[1],None,ms=ms)



P, _, w,intercepts = get_debiasing_projection_with_intercept(SGDClassifier, {}, 1, 2,
                                           True, 0.0, LOCAL_norm[:,0:2],
                                           LOCAL_norm[:,2], LOCAL_norm[:,0:2], LOCAL_norm[:,2],
                                           by_class=False)



###### Display Classifier
params = w[0]

a = -params[0,0]/params[0,1]
xs = np.linspace(0,1,1000)
ys = a*xs - (intercepts[0]/params[0,1])

x1 = (1 + (intercepts[0]/params[0,1]))/a
y1 = a*x1 - (intercepts[0]/params[0,1])

x0 = (0 + (intercepts[0] / params[0, 1])) / a
y0 = a * x0 - (intercepts[0] / params[0, 1])

idx = np.argwhere(np.logical_and(ys>0,ys<1))

xs = xs[idx]
ys = ys[idx]

xs = np.append(xs,x1)
ys = np.append(ys, y1)

xs = np.insert(xs,0,x0)
ys = np.insert(ys,0,y0)
axs[0].plot(xs, ys, 'k',linestyle="dotted",linewidth=15)
axs[0].set_xticks([0.0, 0.5, 1.0])
axs[0].set_yticks([0.0, 0.5, 1.0])
######


axs[0].text(.01, .005, 'a', ha='left', va='bottom',fontsize=50,bbox=dict(facecolor='none', edgecolor='blue'))
axs[1].text(.01, .005, 'b', ha='left', va='bottom',fontsize=50,bbox=dict(facecolor='none', edgecolor='blue'))

LOCAL_norm_projected = LOCAL_norm[:,0:2]@P
LOCAL_norm_projected = scaler.fit_transform(LOCAL_norm_projected)

LOCAL_norm[:,0:2] = LOCAL_norm_projected


ax_new = fig.add_axes([0.223,0.42,0.245,0.54])
plot_synthetic_distribution(LOCAL_norm,ax_new,None,ms=ms)

ax_new.set_xticks([])
ax_new.set_yticks([])

ax_new.text(0.01, .99, 'c', ha='left', va='top',fontsize=50,bbox=dict(facecolor='none', edgecolor='blue'))

plt.tight_layout()
plt.show()