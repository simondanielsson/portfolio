import os
import sys
import time

import numpy as np
import pandas as pd
import random 
import joblib

import autosklearn.classification
from autosklearn.metrics import accuracy, roc_auc, f1, precision, recall

import sklearn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

from smac.facade.roar_facade import ROAR
from smac.scenario.scenario import Scenario

import openml

from pathlib import Path

from cpuinfo import get_cpu_info
from mlens.utils.utils import CMLog

#################################################################
##  Binary classification model evaluation using Auto-sklearn  ##
#################################################################
##
## Binary classification using models from Auto-sklearn.
## Hyperparameter optimization using Auto-sklearn with the search methods GridSearch and RandomSearch. 
## All combinations evaluated on two different datasets.
##
## Author: Simon Danielsson, 
## Contact: simon.danielsson@ericsson.com


def find_feat_types(dataset):
    feat_types = ['Categorical' if feature.name == 'category' else 'Numerical' for feature in dataset['data'].dtypes]
    
    cat_column = []
    for index in range(len(feat_types)):
        if feat_types[index] == 'Categorical':
            cat_column.append(dataset['feature_names'][index])
            
    return feat_types, cat_column


def get_roar_object_callback(scenario_dict, seed, ta, ta_kwargs, backend, metalearning_configurations):
    scenario_dict['input_psmac_dirs'] = backend.get_smac_output_glob()
    scenario = Scenario(scenario_dict)
    
    return ROAR(
        scenario=scenario,
        rng=seed,
        tae_runner=ta,
        tae_runner_kwargs=ta_kwargs,
        run_id=seed,
    )


def get_random_search_object_callback(scenario_dict, seed, ta, ta_kwargs, backend, metalearning_configurations):
    scenario_dict['input_psmac_dirs'] = backend.get_smac_output_glob()
    scenario_dict['minR'] = len(scenario_dict['instances'])
    scenario_dict['initial_incumbent'] = 'RANDOM'
    scenario = Scenario(scenario_dict)
    
    return ROAR(
        scenario=scenario,
        rng=seed,
        tae_runner=ta,
        tae_runner_kwargs=ta_kwargs,
        run_id=seed,
    )


def statistics(dataset, automl, y_train, y_test, y_pred, cm_rss, metric, folds, train_time, search_name, hp_settings): 
    stats = pd.Series()
    
    stats['hp_library'] = 'Autosklearn'
    stats['hp_algorithm'] = search_name
    stats['hp_fix_params'] = hp_settings
    stats['ml_problem_name'] = 'BankMarketingClassification'
    stats['ml_task'] = 'Binary Classification' 
    stats['dataset'] = dataset['details']['name']
    stats['ml_lib'] = 'Sklearn'     
    stats['ml_algorihm'] = [automl.get_models_with_weights()[i][1].get_params()['config'].get('classifier:__choice__')                             for i in range(len(automl.get_models_with_weights()))] 
    stats['ml_fix_params'] = 'Not available'
    stats['hp_search_space'] = 'Not available'
    stats['cv_type'] = 'KFold'
    stats['evaluated_on'] = 'Validation set'
    stats['dataset_size'] = len(dataset['target'])
    stats['k_folds'] = folds
    stats['train_samples_per_fold'] = len(y_train)
    stats['test_samples_per_fold'] = len(y_test)
    stats['train_time'] = train_time
    stats['metric'] = str(metric)
    stats['accuracy'] = autosklearn.metrics.accuracy(y_test, y_pred)
    stats['roc_auc'] = autosklearn.metrics.roc_auc(y_test, y_pred)
    stats['f1'] = autosklearn.metrics.f1(y_test, y_pred)
    stats['precision'] = autosklearn.metrics.precision(y_test, y_pred)
    stats['recall'] = autosklearn.metrics.recall(y_test, y_pred)
    stats['model_type'] = 'Ensemble'
    stats['model_size'] = sys.getsizeof(joblib.dump(automl, 'automl.joblib'))
    stats['avg_pred_time'] = None # TODO: find
    stats['cpu'] = get_cpu_info()['brand_raw']
    stats['cpu_count'] = get_cpu_info()['count']
    stats['gpu_info'] = 'Not compatible'
    stats['gpu_count'] = 'Not compatible'
    stats['ram_usage_avg'] = f"{np.average(np.divide(cm_rss, 1e6)):.3f}" # MB
    
    return stats


def run_pipeline(metric, smac_callable, data_id, time_budget, folds, vanilla):
    # data_id: select dataset with openml id data_id
    # time_budget: max training time for this task

    print_desc = True

    # Start timer for this trial
    start = time.time()   
    
    cm = CMLog(verbose=True)
    cm.monitor(int(time_budget*0.95), ival=5)
    
    # Import data
    print(f"Loading data with id {data_id}...")
    dataset = fetch_openml(data_id=data_id, as_frame=True)
    
    if print_desc:
        print(dataset.DESCR[:445])
    print(f"Dataset '{dataset['details']['name']}' imported")
    
    x, y = dataset['data'], dataset['target']

    # Find feature types for preprocessing
    feat_types, cat_column = find_feat_types(dataset) 
        
    # Convert categorical columns and target labels to int (autosklearn preprocessor can only handle ints) 
    for col in cat_column:
        x.loc[:, col] = LabelEncoder().fit_transform(x[col])
    y = LabelEncoder().fit_transform(y)
    
    random_state = random.randint(0, 10000000)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=random_state)
    print(f"Training set: {len(x_train)} samples,")
    print(f"Validation set: {len(y_test)} samples,") 
    print(f"Val/total ratio: {len(x_test)/len(y)*100:.1f}%")  
    
    # Create automl pipline
    if vanilla:
        automl = autosklearn.classification.AutoSklearnClassifier(ensemble_size=1, initial_configurations_via_metalearning=0)
    else:
        automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=time_budget, resampling_strategy='cv',
                                                              resampling_strategy_arguments={'folds': folds},                                                    
                                                              delete_tmp_folder_after_terminate=False, delete_output_folder_after_terminate=False, 
                                                              get_smac_object_callback=smac_callable[1], metric=metric, n_jobs=-1  
                                      ) # initial_configurations_via_metalearning=0

    # Train models: hp search.
    automl.fit(x_train, y_train, dataset_name=dataset['details']['name'], feat_type=feat_types)
    
    # Print training statistics
    print(f"Training - {automl.sprint_statistics()}")
    
    # Predict validation set labels using optimal model
    x_test_np = x_test.to_numpy()
    y_pred = automl.predict(x_test_np)
    
    # Save model information (incl. performance on validation set)
    cm.collect()
    train_time = time.time() - start
    
    hp_settings = dict()
    if vanilla: 
        hp_settings = {'ensemble_size': 1, 'initial_configurations_via_metalearning': 0}
    else:
        hp_settings = 'Default'
        
    stats = statistics(dataset, automl, y_train, y_test, y_pred, cm.rss, metric, folds, train_time, smac_callable[0], hp_settings)
    
    print(f"Statistics: \n{stats}")
    print('#'*80)
    
    # Export stats to .csv
    local_dir = '/home/edinmsa/results60'
    print(f"Exporting results to {local_dir}...")
    
    time_stop = time.strftime("%Y-%m-%d-%H;%M;%S")
    file_name = f"hpsearch_autosklearn_{dataset['details']['name']}_{smac_callable[0]}_k{folds}_{time_stop}.csv"
    file_dir = Path(local_dir)
    file_dir.mkdir(exist_ok=True)

    pd.DataFrame(stats).to_csv(file_dir / file_name)

    print("Export complete.")
    print('#'*80)
    
    return automl


if __name__ == "__main__":
    start_time = time.time()
    this_time = time.strftime("%H:%M:%S")
    per_task_time_budget = 180 # s (1500s*24 == 10 h)
    
    # Iterate over:
    metrics = [autosklearn.metrics.accuracy, autosklearn.metrics.roc_auc] # Metrics for deciding best model
    smac_callables = {'GridSearch': None, 'RandomSearch': get_random_search_object_callback, 'ROAR': get_roar_object_callback} # Search methods
    data_ids = [1461, 151] # Id's of datasets
    folds = [10, 5] # Number of folds in cv
    vanilla = True # Use vanilla version of autosklearn: no ensemble models and no initial configs. All other settings default

    print(f"Program is going to run for approx.     {np.divide(per_task_time_budget * len(data_ids) * len(metrics) * len(smac_callables) * len(folds), 60*60):.1f} hours.")
    print(f"Program started running {this_time}.")
    
    # Perform experiment
    automl = None # TODO: remove when not performing tests
    for data_id in data_ids:
        for metric in metrics:
            for smac_callable in smac_callables.items(): 
                for fold in folds:
                    automl = run_pipeline(metric, smac_callable, data_id=data_id, time_budget=per_task_time_budget, 
                                          folds=fold, vanilla=vanilla)
    
    time_elapsed = time.time() - start_time    
    print("#"*80)
    print("All iterations completed")
    print(f"Total elapsed time: {time_elapsed:.1f}")
    


