#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 16:25:33 2018

@author: ronaldchao
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()

# extract the data from the data object
features = data['data']
features_label = data['feature_names']
outcome = data['target']
outcome_label = data['target_names']

# training set and test set generation
X_train, X_test, y_train, y_test = train_test_split(features, outcome, test_size = 0.2)

# Define the model
model = tree.DecisionTreeClassifier(max_depth = 6)

# Train the model
model = model.fit(X_train, y_train)

# Run test set
y_test_result = model.predict(X_test)

# Classification results
accuracy = accuracy_score(y_test, y_test_result)

print(accuracy)

from sklearn.externals import joblib
joblib.dump(model, 'model.pkl') 


