#PLEASE WRITE THE GITHUB URL BELOW!
#

import sys
from typing_extensions import ParamSpecArgs
import numpy as np
import pandas as pd
from pandas.io.sql import PandasSQL
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC



def load_dataset(dataset_path):
  data = pd.read_csv(dataset_path)
  return data

def dataset_stat(dataset_df):
  shape = dataset_df.shape
  zero, one = dataset_df.groupby("target").size()
  return shape[1]-1, zero, one

def split_dataset(dataset_df, testset_size):
  X = dataset_df.drop(columns="target", axis=1)
  Y = dataset_df["target"]
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = testset_size, random_state=2)
  return X_train, X_test, Y_train, Y_test


def decision_tree_train_test(x_train, x_test, y_train, y_test):
  dt_cls=DecisionTreeClassifier()
  dt_cls.fit(x_train, y_train)
  accuracy = accuracy_score(y_test, dt_cls.predict(x_test))
  matrix = confusion_matrix(y_test, dt_cls.predict(x_test))
  precision = matrix[1][1] / (matrix[0][1] + matrix[1][1])
  recall = matrix[1][1]/(matrix[1][0] + matrix[1][1])
  return accuracy , precision, recall

def random_forest_train_test(x_train, x_test, y_train, y_test):
  rf_cls=RandomForestClassifier()
  rf_cls.fit(x_train, y_train)
  accuracy = accuracy_score(y_test, rf_cls.predict(x_test))
  matrix = confusion_matrix(y_test, rf_cls.predict(x_test))
  precision = matrix[1][1] / (matrix[0][1] + matrix[1][1])
  recall = matrix[1][1]/(matrix[1][0] + matrix[1][1])
  return accuracy , precision, recall

def svm_train_test(x_train, x_test, y_train, y_test):
  svm_cls = SVC()
  svm_cls.fit(x_train, y_train)
  accuracy = accuracy_score(y_test, svm_cls.predict(x_test))
  matrix = confusion_matrix(y_test, svm_cls.predict(x_test))
  precision = matrix[1][1] / (matrix[0][1] + matrix[1][1])
  recall = matrix[1][1]/(matrix[1][0] + matrix[1][1])
  return accuracy , precision, recall

def print_performances(acc, prec, recall):
    #Do not modify this function!
    print ("Accuracy: ", acc)
    print ("Precision: ", prec)
    print ("Recall: ", recall)

if __name__ == '__main__':
    #Do not modify the main script!
    data_path = sys.argv[1]
    data_df = load_dataset(data_path)

    n_feats, n_class0, n_class1 = dataset_stat(data_df)
    print ("Number of features: ", n_feats)
    print ("Number of class 0 data entries: ", n_class0)
    print ("Number of class 1 data entries: ", n_class1)

    print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
    x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

    acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
    print ("\nDecision Tree Performances")
    print_performances(acc, prec, recall)

    acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
    print ("\nRandom Forest Performances")
    print_performances(acc, prec, recall)

    acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
    print ("\nSVM Performances")
    print_performances(acc, prec, recall)
