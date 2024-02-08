import numpy as np

class DummyClassifier:
    """ Dummy classifier.

               This classifier simply predicts the most often occuring label in Y by calling "fit" and then
               simply predicts the class that occurs most often. This is done to give a baseline of the complexity of the
               dataset as a dummy classifier already performs quite good on some benchmark sets and is by definition always
               fair.
    """
    def __init__(self):
        self.prediction = None

    def fit(self,y):
        self.prediction = np.bincount(y).argmax()

    def predict(self,X):
        sz = np.shape(X)[-1]
        return np.full(sz,self.prediction)