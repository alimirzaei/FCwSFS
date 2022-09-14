import tensorflow as tf
from SequentialChi2 import SequentialChi2

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()


chi2 = SequentialChi2()

chi2.train(X_train=X_train.reshape(X_train.shape[0], -1), y_train=y_train)

accuracy = chi2.test(X_test=X_test, y_test=y_test, max_features= 10)

print("Accuracy = ", accuracy)