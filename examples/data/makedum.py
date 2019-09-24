import csv
import numpy as np
from arch.api import eggroll
from sklearn import linear_model
from sklearn import metrics
import time
from sklearn.model_selection import cross_validate
from math import sqrt

n_samples, n_features = 100, 5
idx = np.array(list(range(n_samples)))
rng = np.random.RandomState(0)
X = rng.randint(low = 0, high = 3, size=(n_samples, n_features))
y = np.dot(X, np.array([1, 2, 1, 1, 2])) + 2
f = round
f = np.vectorize(f)
y = f(y)
X = f(X)
idx = f(idx)
np.savetxt("dum_int_X.csv", X, delimiter=",", fmt='%d')
np.savetxt("dum_int_y.csv", y, delimiter=",", fmt='%d')

X_host = X[...,:3]
data_guest = np.concatenate((idx.reshape(-1, 1), y.reshape(-1, 1), X_host), axis = 1)
data_host = X[...,3:]
data_host = np.concatenate((idx.reshape(-1, 1), data_host), axis = 1)
np.savetxt("dum_int_a.csv", data_host, delimiter=",", fmt='%d')
np.savetxt("dum_int_b.csv", data_guest, delimiter=",", fmt='%d')

# Least Square
linr = linear_model.LinearRegression()
time_start = time.time()
reg = linr.fit(X, y)
time_end = time.time()
duration = (time_end - time_start) * 1000
print("Fitting model takes %.5f" % duration)
predicted = reg.predict(X)

def get_scores(y, predicted):
    scores = []
    scores.append(("explained variance", metrics.explained_variance_score(y, predicted)))
    scores.append(("mean absolute error", metrics.mean_absolute_error(y, predicted)))
    scores.append(("mean squared error", metrics.mean_squared_error(y, predicted)))
    scores.append(("mean squared log error", metrics.mean_squared_log_error(y, predicted)))
    scores.append(("median absolute error", metrics.median_absolute_error(y, predicted)))
    scores.append(("r2 score", metrics.r2_score(y, predicted)))
    scores.append(("root mean squared error", sqrt(metrics.mean_squared_error(y, predicted))))
    return scores

scores = get_scores(y, predicted)
print(scores)

predicted_int = [round(v) for v in predicted]
print("least square")
print(reg.intercept_)
print(reg.coef_)

cv_results = cross_validate(linr, X, y, cv=5,
                            scoring=['explained_variance','neg_mean_absolute_error',
                                     'neg_mean_squared_error','neg_mean_squared_log_error',
                                     'neg_median_absolute_error', 'r2'])
print(cv_results)

# SGDRegressor

clf = linear_model.SGDRegressor(loss='squared_loss', penalty='l2',
                                alpha=0.01, max_iter=400, tol=1e-3,
                                learning_rate='constant', eta0=0.15,
                                shuffle=False, average=100)

cv_results = cross_validate(clf, X, y, cv = 5,  scoring=['explained_variance','neg_mean_absolute_error',
                                     'neg_mean_squared_error','neg_mean_squared_log_error',
                                     'neg_median_absolute_error', 'r2'])


# use ms
print("clf")
time_start = time.time()
clf.fit(X, y)
time_end = time.time()
duration = (time_end - time_start) * 1000
print("Fitting model takes %.5f" % duration)
print(cv_results)
print(clf.intercept_)
print(clf.coef_)
print(clf.n_iter_)




