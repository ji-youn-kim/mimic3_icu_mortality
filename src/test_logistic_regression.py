import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

x_train_logistic_path = "../data/X_train_logistic.npy"
y_train_path = "../data/y_train.npy"
x_test_logistic_path = "../data/X_test_logistic.npy"
y_test_path = "../data/y_test.npy"

x_train_logistic = np.load(x_train_logistic_path)
y_train = np.load(y_train_path)
x_test_logistic = np.load(x_test_logistic_path)
y_test = np.load(y_test_path)

# print(x_train_logistic[:10])
# print(y_train[:10])

log_model = LogisticRegression(solver='lbfgs', max_iter=1000)
clf = log_model.fit(x_train_logistic, y_train.ravel())

train_predictions = clf.predict_proba(x_train_logistic)[:, 1]
test_predictions = clf.predict_proba(x_test_logistic)[:, 1]

train_auroc = roc_auc_score(y_train, train_predictions)
test_auroc = roc_auc_score(y_test, test_predictions)
train_auprc = average_precision_score(y_train, train_predictions)
test_auprc = average_precision_score(y_test, test_predictions)

print("LOGISTIC REGRESSION TRAIN AUROC: ", train_auroc)
print("LOGISTIC REGRESSION TRAIN AUPRC: ", train_auprc)
print("LOGISTIC REGRESSION TEST AUROC: ", test_auroc)
print("LOGISTIC REGRESSION TEST AUPRC: ", test_auprc)

with open('./jiyoun_logistic_regression.txt', 'w') as f:
    f.write(f'{train_auroc:.4f}\n')
    f.write(f'{train_auprc:.4f}\n')
    f.write(f'{test_auroc:.4f}\n')
    f.write(f'{test_auprc:.4f}\n')
