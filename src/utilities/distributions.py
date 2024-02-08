import numpy as np
from aif360.datasets import AdultDataset, CompasDataset

##############################################
# Real World Dataset handling                #
##############################################

def get_dataset(dataset):
    if dataset == 'ADULT':
        data = AdultDataset()

        # 'sex' as sensitive attribute
        prot_attr = data.protected_attribute_names[1]
        prot_idx = data.feature_names.index(prot_attr)

        batch_size = 1000

    elif dataset == 'COMPAS':
        data = CompasDataset()

        prot_attr = data.protected_attribute_names[1]
        prot_idx = data.feature_names.index(prot_attr)

        batch_size = 200

    return data, prot_idx, batch_size


##############################################
# Synthetic Dataset handling                 #
##############################################

def create_synthetic_dataset_XOR(N,seed=None):
    if seed is not None:
        np.random.seed(seed)
    rot_matrix = np.array([[np.cos(np.pi / 2), -np.sin(np.pi / 2)], [np.sin(np.pi / 2), np.cos(np.pi / 2)]])
    # 4 D dataset
    pearls = []

    centers = [0, 0, 1, 1]
    heights = [0, 1, 0, 1]
    prots = [0, 1, 1, 0]

    for i in range(len(centers)):
        pearl = np.transpose(np.hstack((np.random.normal([centers[i]], 0.15, size=(int(N), 1)),
                                        np.random.normal([heights[i]], 0.15, size=(int(N), 1)))))

        # Append protected
        protected = np.ones((1, int(N))) * prots[i]

        pearl = np.concatenate([pearl, protected])

        # Append Label
        y = np.zeros((1, int(N)))

        if heights[i] == 0:
            idx = np.argwhere(pearl[0, :] > 1.1)
            y[:, idx] = 1
        else:
            idx = np.argwhere(pearl[0, :] < 1.1)
            y[:, idx] = 1

        pearl = np.concatenate([pearl, y])

        pearls.append(pearl)

    return np.transpose(np.concatenate(pearls, axis=1))


def create_synthetic_dataset_Local(N,pearl_count,seed=None):
    if seed is not None:
        np.random.seed(seed)

    # 4 D dataset
    pearls = []
    num_pearls = pearl_count

    for i in range(num_pearls):
        if i == 0:
            mn = 6
        else:
            mn = 3

        pearl = np.transpose(np.hstack((np.random.normal([mn], 0.5, size=(int(N), 1)), np.random.normal([0], 0.5, size=(int(N), 1)))))

        # Append protected
        protected = np.zeros((1, int(N)))
        pearl = np.concatenate([pearl, protected])

        # Append Label
        y = np.zeros((1, int(N)))

        if i == 0:
            y = np.zeros((1, int(N)))
        else:
            y = np.ones((1, int(N)))

        pearl = np.concatenate([pearl, y])

        pearls.append(pearl)

    rot_matrix = np.array([[np.cos(np.pi/2),-np.sin(np.pi/2)],[np.sin(np.pi/2),np.cos(np.pi/2)]])

    for i in range(num_pearls):
        if i == 0:
            mn = 6
        else:
            mn = 3

        pearl = np.transpose(np.hstack((np.random.normal([mn], 0.5, size=(int(N), 1)), np.random.normal([0], 0.5, size=(int(N), 1)))))
        pearl = rot_matrix @ pearl

        # Append protected
        protected = np.zeros((1, int(N)))

        idx = np.argwhere(pearl[0, :] < 0)
        protected[:, idx] = 1
        pearl = np.concatenate([pearl, protected])

        # Append Label
        y = np.zeros((1, int(N)))
        idx = np.argwhere(pearl[0, :] < 0)

        y[:, idx] = 1
        pearl = np.concatenate([pearl, y])

        pearls.append(pearl)

    return np.transpose(np.concatenate(pearls,axis=1))

