# Perceptron Multi Capa
from sklearn.neural_network import MLPClassifier
model = MLPClassifier((2,), random_state=0, learning_rate_init=0.1, activation='logistic')
model.fit(X, y)