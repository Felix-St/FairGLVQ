import numpy as np
import multiprocessing

import json
import argparse

from itertools import product
from datetime import datetime

from models.Auxiliary_models import DummyClassifier
from utilities.INP import get_debiasing_projection
from models.GLVQ_models import Fair_GLVQ, GLVQ
from sklearn.linear_model import SGDClassifier

from utilities import distributions
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score
from aif360.sklearn.metrics import statistical_parity_difference, equal_opportunity_difference


def benchmark_dataset(queue, model, dataset, epochs, num_folds, hyperparameter_combinations, generator_entropy):
    result_dictionary = queue.get()
    result_dictionary["dataset"] = dataset

    # Retrieve dataset
    data, prot_idx, batch_size = distributions.get_dataset(dataset)

    X = data.features
    y = data.labels

    nested_dict = {}
    for combination in hyperparameter_combinations:
        # Perform crossvalidation for each hyperparameter
        crossvalidation_result = cross_validate(model,X,y,num_folds,prot_idx,epochs,batch_size,combination,dataset,generator_entropy)

        nested_dict[str(combination)] = crossvalidation_result

    result_dictionary["Parameters"+dataset] = {"BatchSize" : batch_size}
    result_dictionary["results"] = nested_dict
    queue.put(result_dictionary)


def cross_validate(model,X,y,num_folds,prot_idx,epochs,batch_size,combination,dataset_name,generator_entropy):
    seed_generator = np.random.SeedSequence(entropy=generator_entropy)
    seeds = seed_generator.generate_state(num_folds)

    kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=generator_entropy)

    sub_processes = []
    sub_queues = []
    for i, (train_index, test_index) in enumerate(kf.split(X, y)):
        sub_queue = multiprocessing.Queue()

        sp = multiprocessing.Process(target=train_model, args=(sub_queue,model,X, y, train_index, test_index,prot_idx, epochs, batch_size, combination,dataset_name,seeds[i]))

        sp.start()

        sub_processes.append(sp)
        sub_queues.append(sub_queue)

    for sub_process in sub_processes:
        sub_process.join()

    results = {}
    accuracy_results = []
    statistical_parity_results = []
    equal_opportunity_results = []
    for sub_queue in sub_queues:
        evaluation_dictionary = sub_queue.get()
        accuracy_results.append(evaluation_dictionary["Accuracy"])
        statistical_parity_results.append(evaluation_dictionary["StatisticalParity"])
        equal_opportunity_results.append(evaluation_dictionary["EqualOpportunity"])


    results["Accuracy"] = accuracy_results
    results["StatisticalParity"] = statistical_parity_results
    results["EqualOpportunity"] = equal_opportunity_results

    return results


def train_model(sub_queue,model,X,y,train_index,test_index,prot_idx,epochs,batch_size,combination,dataset_name,seed=0):
    # combination[0]: Reg. Param C, combination[1]: Num. Prot., combination[2]: lr
    np.random.seed(seed)

    evaluation_test_data = {}

    X_train = X[train_index]
    y_train = y[train_index]

    X_test = X[test_index]
    y_test = y[test_index]

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if model == "FGLVQ":
        classifier = Fair_GLVQ(X_train, y_train, combination[1], prot_idx, activation="swish", beta=2.0, lr=combination[2], regularization=combination[0],initializer="FAIR_KMEANS")
        classifier.fit(epochs=epochs, batch_size=batch_size)
        y_test_pred = classifier.predict(X_test)
    elif model == "Dummy":
        classifier = DummyClassifier()
        classifier.fit(y_train.reshape(-1, ).astype(int))
        y_test_pred = classifier.predict(np.transpose(X_test))
    elif model == "INP":
        classifier_class = SGDClassifier  # Perceptron

        protected_train = X_train[:, prot_idx]
        protected_test = X_test[:, prot_idx]

        input_dim = np.shape(X_train)[-1]
        is_autoregressive = True
        min_accuracy = 0.0


        num_nullspace_projections = combination[0]

        P, _, _ = get_debiasing_projection(classifier_class, {}, num_nullspace_projections, input_dim,
                                                               is_autoregressive, min_accuracy, X_train,
                                                               protected_train.ravel(), X_test, protected_test.ravel(),
                                                               by_class=False)


        # Projection + normalization of train and test data
        scaler2 = MinMaxScaler()

        X_train_projected = scaler2.fit_transform(X_train @ P)
        X_test_projected = scaler2.transform(X_test @ P)

        # GLVQ
        classifier = GLVQ(X_train_projected, y_train, combination[1], activation="swish", beta=2.0, lr=combination[2],initializer="KMEANS")
        classifier.fit(epochs=epochs, batch_size=batch_size)

        y_test_pred = classifier.predict(X_test_projected)


    accuracy = accuracy_score(y_test, np.asarray(y_test_pred))

    if "COMPAS" in dataset_name:
        equal_opportunity = abs(equal_opportunity_difference(y_test, np.asarray(y_test_pred), pos_label=0, prot_attr=X_test[:, prot_idx]))
        statistical_parity = abs(statistical_parity_difference(y_test, np.asarray(y_test_pred), pos_label=0, prot_attr=X_test[:, prot_idx]))
    else:
        equal_opportunity = abs(equal_opportunity_difference(y_test, np.asarray(y_test_pred), prot_attr=X_test[:, prot_idx]))
        statistical_parity = abs(statistical_parity_difference(y_test, np.asarray(y_test_pred), prot_attr=X_test[:, prot_idx]))

    evaluation_test_data["Accuracy"] = accuracy
    evaluation_test_data["StatisticalParity"] = statistical_parity
    evaluation_test_data["EqualOpportunity"] = equal_opportunity

    sub_queue.put(evaluation_test_data)


def parallel_benchmark(model, datasets, num_folds, epochs, hyperparameter_combinations, generator_entropy):
    processes = []
    queues = []

    # Spawn a process for each dataset, which in turn starts a process for each fold
    # In practice we benchmark each dataset separately due to different regularization values
    for dataset in datasets:
        result_dictionary = {}
        queue = multiprocessing.Queue()
        queue.put(result_dictionary)
        p = multiprocessing.Process(target=benchmark_dataset, args=(queue, model, dataset, epochs, num_folds, hyperparameter_combinations, generator_entropy))
        p.start()
        processes.append(p)
        queues.append(queue)

    for process in processes:
        process.join()

    final_dictionary = {}
    for queue in queues:
        dataset_result_dict = queue.get()
        final_dictionary["Parameters"+dataset_result_dict["dataset"]] = dataset_result_dict["Parameters"+dataset_result_dict["dataset"]]
        final_dictionary[dataset_result_dict["dataset"]] = dataset_result_dict["results"]

    return final_dictionary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='COMPAS',
                        help="(Single) Dataset which should be benchmarked",choices=['COMPAS','ADULT'])
    parser.add_argument('--models', nargs='*', default=['FGLVQ','INP','Dummy'],
                        help="Models which should be benchmarked  ('FGLVQ','INP','Dummy')",choices=['FGLVQ','INP','Dummy'])

    parser.add_argument('--epochs', type=int, help="Number of epochs", default=500)
    parser.add_argument('--folds', type=int, help="Number of cross validation folds", default=5)

    args = parser.parse_args()

    epochs = args.epochs
    num_folds = args.folds
    dataset = [args.dataset]
    models = args.models

    generator_entropy = 4906313 # random.org

    json_name = '.././results/Evaluation_' + str(datetime.now()).replace(":", "_") + ".json"

    models_dictionary = {}
    for model in models:
        if model == "INP":
            if dataset[0] == "COMPAS":
                regularization_parameter = [0, 5, 20, 100, 300, 365, 380]
            elif dataset[0] == "ADULT":
                regularization_parameter = [0, 20, 35, 50, 75, 85, 90, 95]
            number_prototypes = [20]
            learning_rates = [0.05]
        elif model == "FGLVQ":
            if dataset[0] == "COMPAS":
                regularization_parameter = [0, 0.5, 0.75, 1.0, 1.5, 1.75, 2.25]
            elif dataset[0] == "ADULT":
                regularization_parameter = [0, 0.5, 0.75, 1.0, 1.5, 11.0, 12.0, 12.5]
            number_prototypes = [20]
            learning_rates = [0.05]
        elif model == "Dummy": # Const.
            regularization_parameter = [0.0]
            number_prototypes = [0.0]
            learning_rates = [0.0]
        else:
            raise Exception("Unknown model name, use one of {'INP','FGLVQ','Dummy'}")

        hyperparameter_combinations = product(*[regularization_parameter, number_prototypes, learning_rates])

        model_result_dictionary = parallel_benchmark(model, dataset, num_folds, epochs, hyperparameter_combinations, generator_entropy)
        model_result_dictionary["ModelParameters"] = list(hyperparameter_combinations)
        models_dictionary[model] = model_result_dictionary
        models_dictionary["BenchmarkParameters"] = {"NumFolds": num_folds, "Epochs": epochs}

    with open(json_name, 'w') as filepath:
        json.dump(models_dictionary, filepath, indent=4)