# sklearn_logistic_output_summary
It is only suitable for newton-cg method for solver

This package is used to calculate the p value of the coefficient of logical regression in sklearn,
this package is a logistic regression with reference to the satasmodel package, which is the maximum likelihood function estimated by newton-cg method.

This package mainly includes three parts: 
the first part is the statistical value table of model output; 
the second part is the establishment of logistic regression model, including weighted and non weighted parameters; 
the third part is model evaluation, including AUC, KS and related charts and so on.

Please refer to 3.sklearn构建p值.ipynb

example:

import numpy as np

import pandas as pd

from sklearn.datasets import load_boston

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

X, y = load_boston(return_X_y=True)

y_one = [1 if i >20 else 0 for i in y]

y_one = pd.Series(y_one)

X_train, X_test, y_train, y_test = train_test_split(X,y_one,test_size=0.3, random_state=1256)

import sklearn_logistic_model_weight as slmw

sample_weight = [1.5 if i == 1 else 1 for i in y_one]

sample_weight = np.array(sample_weight)

sl = slmw.logistic(X, y_one, X_test, y_test,sample_weight=sample_weight)

res = sl.logistic_()

