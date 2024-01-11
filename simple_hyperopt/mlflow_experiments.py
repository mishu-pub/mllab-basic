# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 1 Data loading and preprocessing
#
# You first need to copy the data from hdfs

# %%
import mlflow
print('The mlflow version is {}.'.format(mlflow.__version__))

import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))

# %%

import numpy as np
import os
import pandas as pd
import scipy
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)

from sklearn.datasets import make_classification



# %% [markdown]
# # For testing purposes
#
# To quickly iterate and see results, we can reduce the training set in the following cell

# %%
number_of_points = 1000

X, y = make_classification(n_samples = number_of_points, random_state=0, weights=[0.5, 0.5],
                           n_features=2000, 
                           n_redundant=10,
                           n_informative = 1800,
                           n_clusters_per_class=2)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0)

# %% [markdown]
# # Search space for hyperopt

# %%

# %%
from hyperopt import Trials, STATUS_OK, tpe, hp, fmin, STATUS_FAIL, space_eval
import numpy as np

space_full = hp.choice('classifiers', [
    {
    'model': LogisticRegression(verbose=0),
    'params': {
        'model__penalty': hp.choice('lr.penalty', ['none', 'l2']),
        'model__C': hp.choice('lr.C', np.arange(0.005,1.0,0.01)),
        'model__solver': hp.choice('lr.solver', [ 'sag']),
        'model__n_jobs': hp.choice('lr.n_jobs', [-1]),

    }
    },
    {
    'model': XGBClassifier(eval_metric='logloss', verbosity=0),
    'params': {
        'model__max_depth' : hp.choice('xgb.max_depth',
                                       range(5, 30, 1)),
        'model__learning_rate' : hp.quniform('xgb.learning_rate',
                                             0.01, 0.5, 0.01),
        'model__n_estimators' : hp.choice('xgb.n_estimators',
                                          range(5, 50, 1)),
        'model__reg_lambda' : hp.uniform ('xgb.reg_lambda', 0,1),
        'model__reg_alpha' : hp.uniform ('xgb.reg_alpha', 0,1)
    }
    },
    #     {
    # 'model': GradientBoostingClassifier(),
    # 'params': {
    #     'model__learning_rate' : hp.quniform('gbc.learning_rate',
    #                                          0.01, 0.5, 0.01),
    #     'model__n_estimators' : hp.choice('gbc.n_estimators',
    #                                       range(5, 100, 1)),
    # }
    # },
    #         {
    # 'model': AdaBoostClassifier(),
    # 'params': {
    #     'model__learning_rate' : hp.quniform('abc.learning_rate',
    #                                          0.01, 0.5, 0.01),
    #     'model__n_estimators' : hp.choice('abc.n_estimators',
    #                                       range(5, 100, 1)),
    # }
    # },
    
])



space = hp.choice('classifiers', [
    {
    'model': LogisticRegression(verbose=0, n_jobs=-1),
    'params': {
        'model__penalty': hp.choice('lr.penalty', [None, 'l2']),
        #'model__C': hp.loguniform('lr.C', np.log(1e-8), np.log(1e-2)),
        'model__C': hp.choice('lr.C', np.arange(0.6, 1.0,0.0001)),
        'model__solver': hp.choice('solver', [ 'sag']),
        'model__max_iter': hp.choice('max_iter', [ 300, 500, 1000]),

    }
    },
])

# %% [markdown]
# # Defining Objective function whose loss we have to minimize
#
#
# We use the log_loss on the test set for that

# %%
from sklearn.metrics import log_loss

def objective(args):
    with mlflow.start_run(nested=True) as child_run:
        # Initialize model pipeline
        model_pipeline = Pipeline(steps=[
            ('model', args['model']) # args[model] will be sent by fmin from search space
        ])

        model_pipeline.set_params(**args['params']) # Model parameters will be set here

        model_pipeline.fit(X_train, y_train)


        loss = log_loss(y_test, model_pipeline.predict_proba(X_test), eps=1e-15)

        print(f"Model Name: {args['model']}: ", loss)

        # Since we have to minimize the score, we return 1- score.
        return {'loss': loss, 'status': STATUS_OK}




# %%
# Hyperopts Trials() records all the model and run artifacts.
trials = Trials()


# %% [markdown]
# # Start the search process with a mlflow parent run

# %%
import mlflow

mlflow.autolog()

with mlflow.start_run() as parent_run:
        
    best_result = fmin(
    fn=objective, 
    space=space,
    algo=tpe.suggest,
    max_evals=3,
    trials=trials)



# %% [markdown]
# # Best_params of the best model

# %%
retrain_best = True
if retrain_best:
    best_params = space_eval(space, best_result)

# %%
if retrain_best:
    best_params

# %%
best_params

# %%
if retrain_best:
    from sklearn.metrics import classification_report
    
    
    model = best_params['model'].fit(X_train, y_train)
    
    # Predicting with the best model
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Classification Report 
    print('Training Classification Report for estimator: ',
          str(model).split('(')[0])
    print('\n', classification_report(y_train, y_pred_train))
    print('\n', classification_report(y_test, y_pred_test))

# %%
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

mlflow.sklearn.autolog()

db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

# Create and train models.
rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
rf.fit(X_train, y_train)

# Use the model to make predictions on the test dataset.
predictions = rf.predict(X_test)


# %%

# %%
