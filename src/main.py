import numpy as np
import options
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

op = options.Options()

x_train_logistic_path = "../data/X_train_logistic.npy"
y_train_path = "../data/y_train.npy"
x_test_logistic_path = "../data/X_test_logistic.npy"
y_test_path = "../data/y_test.npy"

x_train_logistic = np.load(x_train_logistic_path)
y_train = np.load(y_train_path)
x_test_logistic = np.load(x_test_logistic_path)
y_test = np.load(y_test_path)

log_model = LogisticRegression()
log_model.fit(x_train_logistic, y_train.ravel())

predictions = log_model.predict(x_test_logistic)

# print(set(y_train.ravel()) - set(predictions))
# print(set(y_test.ravel()) - set(predictions))

print(classification_report(y_test, predictions))


