import numpy as np
import options
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

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
clf = log_model.fit(x_train_logistic, y_train.ravel())

predictions = clf.predict_proba(x_test_logistic)[:, 1]

log_reg_auroc = roc_auc_score(y_test, predictions)
log_reg_auprc = average_precision_score(y_test, predictions)
# print(classification_report(y_test, predictions))

print("LOGISTIC REGRESSION AUROC: ", log_reg_auroc)
print("LOGISTIC REGRESSION AUPRC: ", log_reg_auprc)


