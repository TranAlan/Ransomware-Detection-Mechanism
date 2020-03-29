#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
    Trains processed file with one class.
    Predict with training data and testing data.
    CSV {Raw features +  discretized + engineered features + Predicted Label + Confidence Score}
    Use Logistic Regression for balanced Confidence Score
'''
from os import makedirs
from os.path import dirname
import sys

import pandas as pd
import numpy as np
import datetime
import time
import pickle

sys.path.append('../')
from utils.file_util import load_yaml

from sklearn.svm import OneClassSVM
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, cross_validate
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, precision_score, recall_score, f1_score
from scipy.stats import uniform, expon


#Global
CONFIG_PATH = './models_config.yml'

def main():
    '''main'''
    config = load_yaml(CONFIG_PATH)
    metric_path = config['metric_path']
    model_path = config['model_path']
    processed_path = config['processed_path']
    trained_path = config['trained_path']
    feature_df = pd.read_csv(processed_path)
    feature_df.loc[feature_df.Label == 0, 'Label'] = -1
    malicious_df = feature_df.loc[feature_df.Label == 1]
    mal_forward_df = malicious_df.loc[malicious_df.is_fwd == 1]
    mal_back_df = malicious_df.loc[malicious_df.is_fwd == 0]
    del malicious_df
    benign_df = feature_df.loc[feature_df.Label == -1]
    benign_size = len(benign_df)
    del feature_df
    X_fwd_train, X_fwd_test, y_fwd_train, y_fwd_test = train_test_split(mal_forward_df, mal_forward_df['Label'], test_size=0.2, random_state=42)
    X_bwd_train, X_bwd_test, y_bwd_train, y_bwd_test = train_test_split(mal_back_df, mal_back_df['Label'], test_size=0.2, random_state=42)

    X_train = pd.concat([X_fwd_train, X_bwd_train])

    X_test = pd.concat([X_fwd_test, X_fwd_test])
    X_test = pd.concat([X_test, benign_df])

    y_train = X_train['Label']
    y_test = X_test['Label']

    del X_fwd_train, X_fwd_test, y_fwd_train, y_fwd_test
    del X_bwd_train, X_bwd_test, y_bwd_train, y_bwd_test
    del benign_df

def get_confusion_matrix(true_label, predict_results):
    #tn, fp, fn, tp = confusion_matrix(true_label, predict_results).ravel()
    return confusion_matrix(true_label, predict_results).ravel().tolist()

if __name__ == '__main__':
    main()
