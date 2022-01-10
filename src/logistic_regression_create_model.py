import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle

x_train_logistic_path = "../data/X_train_logistic.npy"
y_train_path = "../data/y_train.npy"
x_test_logistic_path = "../data/X_test_logistic.npy"
y_test_path = "../data/y_test.npy"
model_save_path = '../data/logistic_regression_model.sav'

x_train_logistic = np.load(x_train_logistic_path)
y_train = np.load(y_train_path)
x_test_logistic = np.load(x_test_logistic_path)
y_test = np.load(y_test_path)

# print(x_train_logistic[:10])
print(y_train[:10])
print(y_test[:10])
print(len(y_test))
print(len(y_train))

log_model = LogisticRegression(solver='lbfgs', max_iter=1000)
log_model.fit(x_train_logistic, y_train)

# save created model
pickle.dump(log_model, open(model_save_path, 'wb'))
