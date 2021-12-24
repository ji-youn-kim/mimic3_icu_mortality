import numpy as np
from torch.utils.data import Dataset, DataLoader

x_train_rnn_path = "../data/X_train_rnn.npy"
y_train_path = "../data/y_train.npy"
x_test_rnn_path = "../data/X_test_rnn.npy"
y_test_path = "../data/y_test.npy"

x_train_logistic = np.load(x_train_rnn_path, allow_pickle=True)
y_train = np.load(y_train_path)
x_test_logistic = np.load(x_test_rnn_path, allow_pickle=True)
y_test = np.load(y_test_path)

# for item in x_train_logistic:
print(x_train_logistic[:10])
print(x_test_logistic[:10])
