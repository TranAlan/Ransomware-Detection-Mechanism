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
import datetime
import time
import pickle
import json

import numpy as np
import pandas as pd

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
    feature_df['StartTime'] = pd.to_datetime(feature_df['StartTime'])
    feature_df.loc[feature_df.Label == 0, 'Label'] = -1
    feature_df.Proto_Int = feature_df.Proto_Int.astype('category')
    feature_df.Sport_Int = feature_df.Sport_Int.astype('category')
    feature_df.Dir_Int = feature_df.Dir_Int.astype('category')
    feature_df.Dport_Int = feature_df.Dport_Int.astype('category')
    feature_df.State_Int = feature_df.State_Int.astype('category')
    malicious_df = feature_df.loc[feature_df.Label == 1]
    mal_forward_df = malicious_df.loc[malicious_df.is_fwd == 1]
    mal_back_df = malicious_df.loc[malicious_df.is_fwd == 0]
    del malicious_df
    benign_df = feature_df.loc[feature_df.Label == -1]
    del feature_df
    X_fwd_train, X_fwd_test, y_fwd_train, y_fwd_test = train_test_split(mal_forward_df,
                                                                        mal_forward_df['Label'],
                                                                        test_size=0.2,
                                                                        random_state=7)
    X_bwd_train, X_bwd_test, y_bwd_train, y_bwd_test = train_test_split(mal_back_df,
                                                                        mal_back_df['Label'],
                                                                        test_size=0.2,
                                                                        random_state=7)

    X_train = pd.concat([X_fwd_train, X_bwd_train])

    X_test = pd.concat([X_fwd_test, X_fwd_test])
    X_test = pd.concat([X_test, benign_df])

    y_train = X_train.Label
    y_test = X_test.Label

    X_train = X_train[:5000]
    y_train = y_train[:5000]

    del X_fwd_train, X_fwd_test, y_fwd_train, y_fwd_test
    del X_bwd_train, X_bwd_test, y_bwd_train, y_bwd_test
    del benign_df

    sample_size = 100000
    if len(X_train) < sample_size:
        sample_size = len(X_train)
    X_train_sample = X_train.sample(100, random_state=7)
    y_train_sample = X_train_sample.Label
    start_time = time.time()
    oc_params = tune_oneclass(df_train_subset(X_train_sample), y_train_sample, 'f1')
    print(f'Time (param search) {sample_size} size. 3 Folds. 18 tot Fits: {time.time()-start_time}')
    oc_kernel = oc_params['kernel']
    oc_nu = oc_params['nu']
    oc_gamma = oc_params['gamma']
    oc_clf = OneClassSVM(kernel=oc_kernel, nu=oc_nu, gamma=oc_gamma, cache_size=7000, verbose=True)
    oc_model_name = 'oneclass_test'
    oc_trained = model_train(oc_clf,
                             X_train,
                             X_test,
                             y_train,
                             y_test,
                             oc_model_name,
                             model_path,
                             metric_path)
    oc_clf = oc_trained['clf']
    oc_scaler = oc_trained['scaler']
    oc_predict_train = oc_trained['predict_train']
    oc_predict_test = oc_trained['predict_test']
    start_time = time.time()
    #Get confidence scores
    data_f = pd.concat([X_train, X_test])
    data_f.sort_values('StartTime', inplace=True)
    oc_conf_score = oc_clf.decision_function(oc_scaler.transform(df_train_subset(data_f)))
    print(f'Time Confidence Scores: {time.time() - start_time}')
    del data_f
    #Saving to CSV
    start_time = time.time()
    x_test_label = X_test['Label']
    X_test.drop(columns=['Label'], inplace=True, axis=1)
    X_test['Label'] = x_test_label
    X_test['Predicted Label'] = oc_predict_test

    mal_train_label = X_train['Label']
    X_train.drop(columns=['Label'], inplace=True, axis=1)
    X_train['Label'] = mal_train_label
    X_train['Predicted Label'] = oc_predict_train

    final_df = pd.concat([X_train, X_test])
    final_df.sort_values('StartTime', inplace=True)

    final_df['Confidence Score'] = oc_conf_score
    makedirs(dirname(f'{trained_path}'), exist_ok=True)
    final_df.to_csv(f'{trained_path}{oc_model_name}_trained.csv', index=False)
    print(f'Saving one_class_featuers csv: {time.time() - start_time}')
    start_time = time.time()
    # Train Logistic Regression


def get_confusion_matrix(true_label, predict_results):
    '''Returns the confusion matrix.
    Args:
        true_label (arr): The true labels of the rows.
        predict_results (arr): The results returned from prediction.
    Returns:
        confusion mattrix: Tuple of size 4 (tn, fp, fn, tp)
    '''
    #tn, fp, fn, tp = confusion_matrix(true_label, predict_results).ravel()
    return confusion_matrix(true_label, predict_results).ravel().tolist()

def df_train_subset(data_f):
    '''
    Returns a copy of the dataframe with columns removed that should
    not be involved with training.
    '''
    col_exclude_training = ['StartTime', 'Dir', 'Proto', 'State', 'Label',
                            'SrcAddr', 'Sport', 'DstAddr', 'Dport', 'sTos', 'dTos', 'is_fwd' ]
    return data_f.drop(columns=col_exclude_training, axis=1).to_numpy()

def fit_predict_model(clf, X, y, scaler_obj):
    '''
    Given X_train and y_train. It will fit the model and
    predict with the trainng data.
    Returns dict containing model and the results from fitting.
    '''
    print('Training Model')
    scaled = scaler_obj.fit(X)
    x_scaled = scaled.transform(X)
    self_predict_r = clf.fit_predict(x_scaled, y=y)
    print('Training Model Completed')
    return {'model': clf, 'self_predict': self_predict_r}

def save_model(model, dir_path, model_name):
    '''
    Saves the model as a pickle
    '''
    print('Saving Model')
    pickle.dump(model, open(f'{dir_path}{model_name}.pickle', 'wb'))

def compare_estimators(clf_list, X, Y, n_jobs=5):
    '''
    Given list of classifiers, it will print the metrics for each of them.
        roc_auc
        Precision = False Positives, at first should be no false positives
        Recall = False Negativives
        f1 =  2 * (precision * recall)/ (precision + recall)
    '''
    # clf_list.append(make_pipeline(preprocessing.StandardScaler(),
    #       LinearSVC(C=27.534917537749216, dual=False, tol=0.0048028537307841352)))
    # clf_list.append(make_pipeline(preprocessing.StandardScaler(),
    #       OneClassSVM(kernel="rbf", gamma=1e-05, cache_size=500, nu=1e-05)))
    scoring = ['accuracy', 'f1', 'recall', 'precision', 'roc_auc']
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    fold = 1
    for clf in clf_list:
        scores = cross_validate(clf, X, Y, scoring=scoring, cv=skf, n_jobs=n_jobs)
        print(scores.keys())
        print(f'----Classifier #{fold}-----')
        print(scores['test_score'])
        fold = fold + 1
        print("Sum Fit Time: %0.5f" % (scores['fit_time'].sum()))
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores['test_accuracy'].mean()*100,
                                               scores['test_accuracy'].std() * 2))
        print("Recall: %0.2f (+/- %0.2f)" % (scores['test_recall'].mean()*100,
                                             scores['test_recall'].std() * 2))
        print("F1: %0.2f (+/- %0.2f)" % (scores['test_f1'].mean()*100,
                                         scores['test_f1'].std() * 2))
        print("Precision: %0.2f (+/- %0.2f)" % (scores['test_precision'].mean()*100,
                                                scores['test_precision'].std() * 2))
        print("ROC: %0.2f (+/- %0.2f)" % (scores['test_roc_auc'].mean()*100,
                                          scores['test_roc_auc'].std() * 2))
        print()

def model_performance_metrics(y_true, y_pred):
    '''
    Returns a dictionary for the metrics of a given test result.
    '''
    metric_results_dict = {}
    metric_results_dict['accuracy'] = accuracy_score(y_true, y_pred)
    metric_results_dict['recall'] = recall_score(y_true, y_pred, average='binary')
    metric_results_dict['precision'] = precision_score(y_true, y_pred, average='binary')
    metric_results_dict['f1'] = f1_score(y_true, y_pred, average='binary')
    metric_results_dict['average_precision'] = average_precision_score(y_true, y_pred)
    metric_results_dict['confusion_matrix'] = get_confusion_matrix(y_true, y_pred)
    return metric_results_dict

def hyper_tuning(classifier, tuned_parameters, X, y, score):
    '''
        Perform GridSearch Cross Validate to find best paramers given the score metric.
        Tuned Paramters is a dictionary with options for each parameter
        Ex Metrics.
            score = 'roc_auc'
            score = 'f1'
            score = 'accuracy'
    '''
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(
        classifier, tuned_parameters, scoring=score, n_jobs=5, cv=3, verbose=25
    )
    scaler = preprocessing.StandardScaler().fit(X)
    clf.fit(scaler.transform(X), y)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()
    return clf.best_params_

def tune_oneclass(X, y, score):
    '''
    Returns the best parameters for the model
    with respect the scoring function.
    '''
    #     tuned_parameters = [{'kernel': ['rbf'],
    #                          'gamma': expon(scale=.1),
    #                          'nu': expon(scale=0.1)}]
    tuned_parameters = [{'kernel': ['rbf'],
                         'gamma': [1e-6, 1e-5, 1e-4],
                         'nu': [1e-3, 1e-2, 0.1]}]
    return hyper_tuning(OneClassSVM(), tuned_parameters, X, y, score)

def model_train(clf, X_train, X_test, y_train, y_test, model_name, model_path, metric_path):
    '''
    Fit and Predicts model on train data.
    Saves training metrics and confusion matrix.
    Predicts with testing data.
    Saves testing metrics and confusion matrix.
    Creates CSV with Predicted Labels and Confidence Score.
    '''
    scaler = preprocessing.StandardScaler()
    start_time = time.time()
    modeling_dict = fit_predict_model(clf, df_train_subset(X_train), y_train, scaler)
    print(f'Time Fiting and Predicting train data: {time.time() - start_time}')
    clf = modeling_dict['model']
    save_model(clf, model_path, model_name)
    self_predict_r = modeling_dict['self_predict']
    train_performance = model_performance_metrics(y_train, self_predict_r)

    print(train_performance)
    print()

    df_confusion_train = pd.crosstab(y_train,
                                     self_predict_r,
                                     rownames=['Actual'],
                                     colnames=['Predicted'],
                                     margins=True)
    df_confusion_train_norm = pd.crosstab(y_train,
                                          self_predict_r,
                                          rownames=['Actual'],
                                          colnames=['Predicted'],
                                          normalize='index')

    print(df_confusion_train)
    print()
    print(df_confusion_train_norm)
    print()
    with open(f'{metric_path}{model_name}_train_matrix.txt', 'w') as outfile: 
        outfile.write(df_confusion_train.to_string())
    with open(f'{metric_path}{model_name}_train_matrix_norm.txt', 'w') as outfile: 
        outfile.write(df_confusion_train_norm.to_string())

    #Save Metrics
    makedirs(dirname(metric_path), exist_ok=True)
    with open(f'{metric_path}{model_name}_train_metric.json', 'w') as outfile: 
        outfile.write(json.dumps(train_performance, indent = 4) )
    testing_results = None

    if X_test is not None and y_test is not None:
        start_time = time.time()
        testing_results = clf.predict(scaler.transform(df_train_subset(X_test)))
        print(f'Time testing with testing data: {time.time() - start_time}')
        test_performance = model_performance_metrics(y_test, testing_results)
        print(test_performance)
        with open(f'{metric_path}{model_name}_test_metric.json', 'w') as outfile:
            outfile.write(json.dumps(train_performance, indent=4))

        df_confusion_test = pd.crosstab(y_test,
                                        testing_results,
                                        rownames=['Actual'],
                                        colnames=['Predicted'],
                                        margins=True)
        df_confusion_test_norm = pd.crosstab(y_test,
                                             testing_results, rownames=['Actual'],
                                             colnames=['Predicted'],
                                             normalize='index')
        print(df_confusion_test)
        print()
        print(df_confusion_test_norm)
        print()
        with open(f'{metric_path}{model_name}_test_matrix.txt', 'w') as outfile:
            outfile.write(df_confusion_test.to_string())
        with open(f'{metric_path}{model_name}_test_matrix_norm.txt', 'w') as outfile:
            outfile.write(df_confusion_test_norm.to_string())
    return {
        'clf': clf,
        'scaler': scaler,
        'predict_train': self_predict_r,
        'predict_test': testing_results
    }

if __name__ == '__main__':
    main()
