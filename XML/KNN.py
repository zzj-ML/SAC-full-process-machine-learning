#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn import metrics
from sklearn.neighbors import KNeighborsRegressor
from joblib import dump,load
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


df = pd.read_csv('...')

X = df.drop(columns=['RDFT'])
Y = df['RDFT']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)
print("X_train's shape is", X_train.shape,"; y_train's shape is", y_train.shape)
print("X_test's shape is", X_test.shape,"; y_test's shape is",y_test.shape)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_stand = scaler.transform(X_train)
X_test_stand = scaler.transform(X_test)


param_grid = {
    'weights': ['uniform', 'distance'],
    'n_neighbors': range(1, 20),
    'algorithm': ['ball_tree', 'kd_tree', 'brute'],
}

knn_regressor = KNeighborsRegressor()

grid_search = GridSearchCV(estimator=knn_regressor, param_grid=param_grid, cv=5, verbose=4, n_jobs=-1)

grid_search.fit(X_train_stand, y_train)
best_model = grid_search.best_estimator_
print("Best parameters found:", grid_search.best_params_)

y_train_hat = best_model.predict(X_train_stand)
y_test_hat = best_model.predict(X_test_stand)


fontsize = 22
plt.figure(figsize=(8, 6))

a = plt.scatter(y_train, y_train_hat, s=150, c=[(104/255, 136/255, 245/255)], label='Train')

b = plt.scatter(y_test, y_test_hat, s=150, c=[(215/255, 112/255, 113/255)], label='Test')

plt.plot([1.5, 6.3], [1.5, 6.3], color='k', linestyle='--')

plt.xlabel('Experimental value of RFDT', fontsize=fontsize)
plt.ylabel('ML predicted value of RFDT', fontsize=fontsize)

r2 = r2_score(y_test, y_test_hat)
mae = mean_absolute_error(y_test, y_test_hat)
rmse = np.sqrt(mean_squared_error(y_test, y_test_hat))

plt.text(2.37, 5.75, f' MAE={mae:.2f}, RMSE={rmse:.2f}', fontsize=24)

plt.xlim([1.5, 6.3])
plt.ylim([1.5, 6.3])

plt.legend(fontsize=18, handletextpad=0.1, borderpad=0.1)
plt.tick_params(axis='both', which='major', labelsize=fontsize)

plt.tight_layout()

# plt.savefig('knn_pybel.png', dpi = 1000)
plt.show()






