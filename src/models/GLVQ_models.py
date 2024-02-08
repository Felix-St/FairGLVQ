from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

import numpy as np
import math

def sigmoid(x,beta=1.0):
  return 1 / (1 + np.exp(-beta*x))

def swish(x,beta=1.0):
    return x*sigmoid(x,beta)

def sigmoid_der(x,beta=1.0):
    return beta*sigmoid(x,beta)*(1-sigmoid(x,beta))

def swish_der(x,beta=1.0):
    return beta*swish(x,beta)+sigmoid(x,beta)*(1-beta*swish(x,beta))

class GLVQ():

    def __init__(self,X,y,num_prototypes_per_class,lr=0.01,seed=None,initializer="KMEANS",activation="swish",beta=2.0):
        """ Initializes GLVQ class.

            Parameters
            ----------
            X : array of shape (n_samples,n_features)
                Training Data
            y : array of shape (n_samples,)
                Labels
            num_prototypes_per_class : int
                number of prototypes used per class
            lr : float
                learning rate eta
            seed : int
                seed for random elements in initialization of prototypes for reproducibility
            initializer : string, {"MPC","MEAN","KMEANS","FAIR_KMEANS"}
                - "MPC"/"MEAN": initialized in center of each class
                - "KMEANS" initialized using class centers of per class kmeans clustering
                - "FAIR_KMEANS" adjusted kmeans, 2 prototypes for each class center in clustering across all classes w/ noise
            activation : string, {"swish","sigmoid"}
                specifies if swish or sigmoid activation should be used
            beta : base
                beta parameter for scaling the activation function

        """

        self.alpha = 2 # Used if no correct/wrong prototype exists (see FairGLVQ)

        self.N = np.shape(X)[0]
        self.num_classes = len(np.unique(y))
        self.X = X
        self.y = y
        self.num_prototypes_per_class = num_prototypes_per_class
        self.seed = seed

        self.dimension = np.shape(X)[-1]
        self.initializer = initializer  # MPC: mean per class, MEAN: mean of whole dataset

        self.prototypes = self.initialize_prototypes(self.num_classes, num_prototypes_per_class)

        self.lr = lr

        self.beta = beta
        self.activation = activation


    def fit(self,epochs,batch_size=1,verbose=10):
        """ Trains the GLVQ model.

            Parameters
            ----------
            epochs : int
                Number of epochs to train
            batch_size:
                Batch size for speeding up computation
            verbose : int / bool
                How often the current epoch should be printed (False for no display)

        """
        for e in range(epochs):
            if verbose != False:
                if e % verbose == 0:
                    print("=== GLVQ Epoch: ", e)

            # For every batch
            for i in range(math.ceil(self.N/batch_size)):
                if (i+1)*batch_size < self.N:
                    batch = self.X[i*batch_size:(i+1)*batch_size]
                    batch_labels = self.y[i*batch_size:(i+1)*batch_size]
                else:
                    batch = self.X[i * batch_size:]
                    batch_labels = self.y[i * batch_size:]

                prototype_classes = self.get_prototype_labels()
                accumulated_gradients = self._compute_GLVQ_update(batch,batch_labels,prototype_classes)

                # At the end of every batch update the prototypes with the accumulated gradients
                for i,prototype in enumerate(self.prototypes):
                    prototype.update(self.lr*accumulated_gradients[:,i])



    def _compute_GLVQ_update(self,batch,batch_labels,prototype_attributes):
        accumulated_gradients = np.zeros((self.dimension, len(self.prototypes)))
        prototypes_numpy = np.asarray(self.prototypes)
        number_of_data_used = 0
        for j in np.unique(batch_labels):
            batch_correct, correct_prototypes, wrong_prototypes, original_indices_correct, original_indices_wrong = self.__organise_data(batch, batch_labels, prototypes_numpy, prototype_attributes, j)

            number_of_data_used += np.shape(correct_prototypes)[0] + np.shape(wrong_prototypes)[0]

            if np.shape(correct_prototypes)[0] == 0:
                closest_prototypes_wrong, closest_distances_wrong, differences_wrong = self.__compute_aggregated_distances(batch_correct, wrong_prototypes)
                closest_distances_correct = np.multiply(closest_distances_wrong, self.alpha)
            elif np.shape(wrong_prototypes)[0] == 0:
                closest_prototypes_correct, closest_distances_correct, differences_correct = self.__compute_aggregated_distances(batch_correct, correct_prototypes)
                closest_distances_wrong = np.divide(closest_distances_correct, self.alpha)
            else:
                closest_prototypes_wrong, closest_distances_wrong, differences_wrong = self.__compute_aggregated_distances(batch_correct, wrong_prototypes)
                closest_prototypes_correct, closest_distances_correct, differences_correct = self.__compute_aggregated_distances(batch_correct, correct_prototypes)

            # Compute the update, we computed w1,d1,w2,d2 etc. for the whole batch now
            mu_x = np.divide(np.subtract(closest_distances_correct, closest_distances_wrong), np.add(closest_distances_correct, closest_distances_wrong))

            if self.activation == "sigmoid":
                prefactor = sigmoid_der(mu_x,self.beta)
            elif self.activation == "swish":
                prefactor = swish_der(mu_x,self.beta)
            else:
                raise("Please provide valid activation function (valid: 'swish','sigmoid'")

            if np.shape(correct_prototypes)[0] != 0:
                # Closest Prototypes with the same class
                scaling_correct = np.divide(closest_distances_wrong,np.power(np.add(closest_distances_correct, closest_distances_wrong), 2))
                non_aggregated_update_correct = (prefactor * scaling_correct)[:, None] * differences_correct

                for prototype_index in np.unique(closest_prototypes_correct):
                    mask = (closest_prototypes_correct == prototype_index)
                    accumulated_gradients[:, original_indices_correct[prototype_index]] += np.sum(non_aggregated_update_correct[mask], axis=0)

            if np.shape(wrong_prototypes)[0] != 0:
                # Closest Prototypes with a different class
                scaling_wrong = np.divide(closest_distances_correct,np.power(np.add(closest_distances_correct, closest_distances_wrong), 2))
                non_aggregated_update_wrong = (prefactor * scaling_wrong)[:, None] * differences_wrong

                for prototype_index in np.unique(closest_prototypes_wrong):
                    mask = (closest_prototypes_wrong == prototype_index)
                    accumulated_gradients[:, original_indices_wrong[prototype_index]] -= np.sum(non_aggregated_update_wrong[mask], axis=0)

        if number_of_data_used != 0:
            return np.divide(accumulated_gradients,number_of_data_used)
        else:
            return np.divide(accumulated_gradients,1.0)


    def __compute_aggregated_distances(self,batch,prototypes):
        distances = np.asarray([np.power(np.linalg.norm(batch - prototype[:], axis=1),2) for prototype in prototypes])
        closest_prototypes = np.argmin(distances, axis=0)
        closest_distances = np.min(distances, axis=0)

        temp_correct = [prototype[:] for prototype in list(prototypes[closest_prototypes])]
        differences = np.subtract(batch, temp_correct)

        return closest_prototypes, closest_distances, differences

    def __organise_data(self,batch,batch_labels,prototypes,prototype_classes,current_class):
        index_data = (batch_labels == current_class).reshape(-1, )
        batch_correct = batch[index_data]

        index = (prototype_classes == current_class)

        correct_prototypes = prototypes[index]
        wrong_prototypes = prototypes[~index]

        original_indices_correct = np.arange(0, len(prototypes))[index]
        original_indices_wrong = np.arange(0, len(prototypes))[~index]

        return batch_correct, correct_prototypes, wrong_prototypes, original_indices_correct, original_indices_wrong


    def _get_closest_prototype(self,x):
        minimum_distance = np.inf
        for i in range(len(self.prototypes)):
            distance = np.linalg.norm(x - self.prototypes[i][:])

            if distance < minimum_distance:
                closest_prototype = self.prototypes[i]
                minimum_distance = distance
        return closest_prototype

    def initialize_prototypes(self,num_classes,num_prot_per_class):
        if self.seed is not None:
            np.random.seed(self.seed)

        prototypes = []
        if self.initializer=="MPC" or self.initializer=="MEAN":
            mean_per_class = self.__compute_initialization_mean()
            for i in range(num_classes):
                for j in range(num_prot_per_class):
                    prototypes.append(Prototype(np.random.multivariate_normal(mean_per_class[i],np.eye(self.dimension)*0.005),i))
        elif self.initializer == "KMEANS":
            # We here use KMEANS for each class separately
            for i in range(num_classes):
                idx = np.ravel(np.argwhere(np.ravel(self.y) == i))
                Xi = self.X[idx,:]
                with threadpool_limits(limits=1, user_api='openmp'):
                    kmeans = KMeans(n_clusters=num_prot_per_class, n_init="auto", random_state=self.seed).fit(Xi)

                for center in kmeans.cluster_centers_:
                    prototypes.append(Prototype(center, i))

        elif self.initializer == "FAIR_KMEANS":
            # KMEANS across classes then initialize num_classes of prototypes per cluster center
            with threadpool_limits(limits=1, user_api='openmp'):
                kmeans = KMeans(n_clusters=num_prot_per_class, n_init="auto", random_state=self.seed).fit(self.X)

            for i in range(num_classes):
                for center in kmeans.cluster_centers_:
                    prototypes.append(Prototype(center+np.random.multivariate_normal(np.zeros(self.dimension),np.eye(self.dimension)*0.0005), i))

        else:
            raise Exception('Unknown Initializer, use one of: [MPC,MEAN,KMEANS,FAIR_KMEANS]')
        return prototypes

    def __compute_initialization_mean(self):
        if self.initializer=="MPC":
            return [np.mean(self.X[(self.y==label).reshape(-1,),:],axis=0) for label in np.unique(self.y)]
        elif self.initializer=="MEAN":
            return [np.mean(self.X, axis=0) for label in np.unique(self.y)]
        else:
            raise("Please provide a valid initializer (valid: MPC, MEAN)")


    def get_prototypes(self):
        return self.prototypes

    def get_prototype_labels(self):
        return np.asarray([prototype.get_label() for prototype in self.prototypes])

    def get_prototype_positions(self):
        positions = np.zeros((self.dimension,len(self.prototypes)))
        for i, prototype in enumerate(self.prototypes):
            positions[:,i] = prototype[:]
        return positions

    def predict(self,X):
        """ Predict on new data X.

           Parameters
           ----------
           X : array of shape (n_samples,n_features)
               New Data to predict on

           Returns
           -------
           list of length n_samples
                Predicted class for data X

        """
        N = np.shape(X)[0]
        classes = []
        for i in range(N):
            closest_prototype = self._get_closest_prototype(X[i])
            classes.append(closest_prototype.get_label())
        return classes


class Fair_GLVQ(GLVQ):
    def __init__(self,X,y,num_prototypes_per_class,protected_attribute_index,lr=0.01,seed=None,initializer="FAIR_KMEANS",activation="swish",beta=2.0,regularization=0,use_protected_attribute=True):
        """ Initializes FairGLVQ class.
            Parameters
            ----------
            X : array of shape (n_samples,n_features)
                Training Data
            y : array of shape (n_samples,)
                Labels
            num_prototypes_per_class : int
                number of prototypes used per class
            protected_attribute_index : int
                index of the protected attribute, e.g. X[:,protected_attribute_index] would be the column vector
            lr : float
                learning rate eta
            seed : int
                seed for random elements in initialization of prototypes for reproducibility
            initializer : string, {"MPC","MEAN","KMEANS","FAIR_KMEANS"}
                - "MPC"/"MEAN": initialized in center of each class
                - "KMEANS" initialized using class centers of per class kmeans clustering
                - "FAIR_KMEANS" adjusted kmeans, 2 prototypes for each class center in clustering across all classes w/ noise
            activation : string, {"swish","sigmoid"}
                specifies if swish or sigmoid activation should be used
            beta : base
                beta parameter for scaling the activation function
            regularization  : float
                fairness regularization parameter
            use_protected_attribute : bool
                boolean to indicate if the protected attribute should be considered part of the data or not
        """

        # Decide if the protected attribute should be used for classification itself or not
        if not use_protected_attribute:
            super().__init__(np.delete(X,protected_attribute_index,1),y,num_prototypes_per_class,lr,seed,initializer,activation,beta)
        else:
            super().__init__(X, y, num_prototypes_per_class, lr, seed, initializer, activation, beta)

        self.protected_attribute = X[:, protected_attribute_index]

        self.regularization = regularization

        self.add_sensitive_to_prototypes()
        self.update_sensitive_assignment()


    def fit(self,epochs,batch_size=1,verbose=10):
        num_batches = math.ceil(self.N / batch_size)
        for e in range(epochs):
            if verbose != False:
                if e % verbose == 0:
                    print("=== FairGLVQ Epoch: ", e)
            # For every batch
            for i in range(num_batches):
                if (i + 1) * batch_size < self.N:
                    batch = self.X[i * batch_size:(i + 1) * batch_size]
                    batch_labels = self.y[i * batch_size:(i + 1) * batch_size]
                    batch_sensitive = self.protected_attribute[i * batch_size:(i + 1) * batch_size]
                else:
                    batch = self.X[i * batch_size:]
                    batch_labels = self.y[i * batch_size:]
                    batch_sensitive = self.protected_attribute[i * batch_size:]

                prototype_classes = self.get_prototype_labels()
                prototype_sensitives = self.get_prototype_sensitives()

                accumulated_GLVQ_gradients = self._compute_GLVQ_update(batch, batch_labels, prototype_classes)
                accumulated_FAIR_gradients = self._compute_GLVQ_update(batch, batch_sensitive, prototype_sensitives)

                accumulated_gradients = accumulated_GLVQ_gradients - self.regularization*accumulated_FAIR_gradients

                # At the end of every batch update the prototypes with the accumulated gradients
                for k, prototype in enumerate(self.prototypes):
                    prototype.update(self.lr*accumulated_gradients[:, k])

                # Update the pseudo-classes assigned to the prototypes for fairness
                # Note that this might take a long time for large datasets and large prototype counts
                self.update_sensitive_assignment()


    def update_sensitive_assignment(self):

        distances = np.zeros((len(self.prototypes),np.shape(self.X)[0]))
        for i,prototype in enumerate(self.prototypes):
            distances[i,:] = np.linalg.norm(self.X-prototype[:],axis=1)

        """ Vectorized Alternative
            distances = np.linalg.norm(np.reshape(self.X, [-1, self.dimension]) - np.reshape(np.array([prototype[:] for prototype in self.prototypes]), [-1, 1, self.dimension]), axis=2)
        """

        """ Threaded Alternative
            distances = Parallel(n_jobs=-1, backend="threading")(
            delayed(np.linalg.norm)(self.X - prototype[:], axis=1) for prototype in self.prototypes)
            distances = np.asarray(distances)
        """

        closest_prototypes_indices = np.argmin(distances,axis=0)

        unique_protected = np.unique(self.protected_attribute)

        arrangements = np.zeros((len(np.unique(self.protected_attribute)),len(self.prototypes)))
        for i,protected in enumerate(np.unique(self.protected_attribute)):
            idx = np.where(self.protected_attribute == protected)
            counts = np.bincount(closest_prototypes_indices[idx],minlength=len(self.prototypes))
            arrangements[i,:] = counts


        # This ensures that even if the protected attribute are not all present this will work
        # e.g., [1 4] instead of [0,1]
        idx = np.argmax(arrangements, axis=0)
        sensitive_labels = unique_protected[idx]

        for i,prototype in enumerate(self.prototypes):
            prototype.add_sensitive(sensitive_labels[i])


    def add_sensitive_to_prototypes(self):
        for prototype in self.prototypes:
            prototype.add_unique_sensitives(np.unique(self.protected_attribute))

    def get_prototype_sensitives(self):
        return np.asarray([prototype.get_sensitive() for prototype in self.prototypes])

class Prototype():
    def __init__(self,position,label):
        self.position = position
        self.label = label
        self.sensitive = None

    def add_unique_sensitives(self,number_of_sensitives):
        self.unique_sensitives = number_of_sensitives

    def get_distance(self,x):
        return np.linalg.norm(x-self.position)

    def get_label(self):
        return self.label

    def get_sensitive(self):
        return self.sensitive

    def add_sensitive(self,value):
        self.sensitive = value

    def update(self,x):
        self.position = self.position + x

    def __getitem__(self,idx):
        return self.position[idx]


