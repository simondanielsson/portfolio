import os
import sys
import time
import random
import numpy as np
import pandas as pd
import scipy
import joblib

import ray
from ray import tune
from ray.tune import run, Experiment
from ray.tune import grid_search
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import FIFOScheduler, ASHAScheduler

import xgboost as xgb
from xgboost.sklearn import XGBClassifier

from hyperopt import hp

import sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier

from pathlib import Path

from cpuinfo import get_cpu_info
import psutil


#################################################################
##    Binary classification model evaluation using Ray Tune    ##
#################################################################
##
## Binary classification using Gradient Boosted Decision Trees (GBDT) from XGBoost and Sklearn.
## Hyperparameter optimization using Ray Tune with the search methods GridSearch, RandomSearch, 
## and HyperOptSearch. All combinations evaluated on two different datasets.
##
## Author: Simon Danielsson, 
## Contact: simon.danielsson@ericsson.com




def get_ml_alg(ml_lib):
    # Get information about estimator 
    if ml_lib == 'XGBoost':
        return 'gbtree', 'Ensemble (boosting)'
    return 'gbdt', 'Ensemble (boosting)'
    

def statistics(clf, dataset, ml_lib, y_train, y_test, y_pred, metric,                search_name, config, search_space_ends, search_settings, fix_params, experiment_dir):
    ## Save all statistics to Series 
    
    # Get details about ml algorithm used in ml_lib
    ml_alg, alg_type = get_ml_alg(ml_lib)
    
    # Get hyperparameters space. When using other search methods than GridSearch, 
    # only the boundary of the search space can be returned
    search_space = dict()
    if search_name == 'GridSearch':
        for key, value in config['hyperparams'].items():
            if type(value) is dict: 
                search_space[key] = list(value.values())[0]
    else: 
        search_space = search_space_ends 
    
    # Compute average ram usage and training time to get the best model
    trials = tune.Analysis(experiment_dir)

    tot_ram_usage = []
    time_trials = []
    for trial, values in trials.trial_dataframes.items():
        tot_ram_usage.append(values['avg_memory_usage'].values[0])
        time_trials.append(values['time_total_s'].values[0])
        
    avg_ram_usage = np.average(tot_ram_usage)
    tot_time = np.sum(time_trials)
    
    # Add all information
    stats = pd.Series()
    stats['hp_library'] = 'Ray Tune'
    stats['hp_algorithm'] = search_name
    stats['hp_fix_params'] = search_settings
    stats['ml_problem_name'] = f"{dataset['details']['name']}Classification"
    stats['ml_task'] = 'Binary Classification' 
    stats['dataset'] = dataset['details']['name']
    stats['ml_lib'] = ml_lib     
    stats['ml_algorithm'] = ml_alg
    stats['ml_fix_params'] = fix_params
    stats['hp_search_space'] = search_space
    stats['cv_type'] = 'KFold'
    stats['evaluated_on'] = 'Validation set'
    stats['dataset_size'] = len(dataset['target'])
    stats['k_folds'] = '10'
    stats['train_samples_per_fold'] = len(y_train)
    stats['test_samples_per_fold'] = len(y_test)
    stats['train_time'] = tot_time
    stats['metric'] = metric
    stats = stats.append(pd.Series(evaluate(y_test, y_pred)))
    stats['model_type'] = alg_type
    stats['model_size'] = sys.getsizeof(joblib.dump(clf, 'clf.joblib'))
    stats['avg_pred_time'] = None # TODO: find
    stats['cpu'] = get_cpu_info()['brand_raw']
    stats['cpu_count'] = get_cpu_info()['count']
    stats['gpu_info'] = None
    stats['gpu_count'] = None
    stats['ram_usage_avg'] = avg_ram_usage
    
    return stats


def evaluate(y_test, y_pred):
    # Evaluate classifier predicitions on some data
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    precision = 0 if tp + fp == 0 else np.divide(tp, float(tp + fp))
    recall = 0 if tp + fn == 0 else np.divide(tp, float(tp + fn))
    
    # Add results to list
    df = dict()
    df["accuracy"] = np.divide(tp + tn, float(tp + tn + fp + fn))
    df["precision"] = precision
    df["recall"] = recall
    df["f1"] = 0 if precision + recall == 0 else np.divide(2 * precision * recall, precision + recall)
    df["selectivity"] = 0 if tn + fp == 0 else np.divide(tn, float(tn + fp))
    df["roc_auc"] = roc_auc_score(y_test, y_pred)

    return df


def get_clf(ml_lib, hyperparams): 
    # Get classifier
    if ml_lib == 'XGBoost': 
        return XGBClassifier(**hyperparams)
    return GradientBoostingClassifier(**hyperparams)
    

class TrainableModel(tune.Trainable): #subclass to tune.Trainable: used in tune.run()

    def _setup(self, config):
        self.x = ray.get(self.config['x_id']) # Gain access to data
        self.y = pd.DataFrame(ray.get(self.config['y_id']))
        self.ml_lib = self.config['ml_lib']
        self.search_alg = self.config['search_alg']
        self.hyperparams = self.config['hyperparams']
        
    def _train(self):
        start_train = time.time()
        
        # Track this process (memory)
        process = psutil.Process(os.getpid()) 
 
        cv_scores = [] 
        memory_usage = []
        results = dict()
        
        n_splits = 10 #n_splits-fold cross validation
        random_state = random.randint(0, 10000)
        k_fold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        # Training: cv
        for train_index, validation_index in k_fold.split(self.x, self.y): 

            # Split data
            x_train, x_validation = self.x.values[train_index], self.x.values[validation_index]
            y_train, y_validation = self.y.values[train_index], self.y.values[validation_index]

            # Train classifier
            clf = get_clf(self.ml_lib, self.hyperparams).fit(x_train, y_train)
            
            # Track memory usage
            memory_usage.append(process.memory_info().rss)
            
            # Predict labels
            y_pred= clf.predict(x_validation)

            # Calculate metrics for this iteration of cv
            cv_scores.append(evaluate(y_validation, y_pred))
                        
        # Calculate average across all folds of all metrics 
        for key in cv_scores[0].keys():
            results[key] = np.mean([column[key] for column in cv_scores])
        
        avg_memory_usage = np.divide(np.average(memory_usage), 1e6) # convert unit to MB
        
        results['avg_memory_usage'] = avg_memory_usage
        results['train_time'] = time.time() - start_train
    
        return results
    

def get_config(search_name, ml_lib, x_id, y_id, space_boundary, n_jobs=-1):
    # GridSearch and RandomSearch: Get all hyperparameters. 
    # HyperOptSearch and BayesOptSearch: Get fix (default) parameters 
    config = {
        'x_id': x_id,
        'y_id': y_id,
        'search_alg': search_alg,
        'ml_lib': ml_lib
    } 
    hyperparams = dict()
    
    # Get hyperparams and config for Grid- and RandomSearch
    if ml_lib == 'XGBoost':
        fix_params = {
                'n_jobs': n_jobs,
                'verbosity': 0,
                'booster': 'gbtree',
                'min_child_weight': 1,
                'max_delta_step': 0, 
                'subsample': 1,
                'sampling_method': 'uniform',
                'lambda': 1,
                'scale_pos_weight': 1
        }
        if search_name == 'GridSearch':
             hyperparams = {
                'max_depth': grid_search(list(np.linspace(space_boundary['max_depth'][0],  # DEFAULT: 6
                                                          space_boundary['max_depth'][1], num=SAMPLE_SIZE, dtype=int))),
                'learning_rate': grid_search(list(np.linspace(space_boundary['learning_rate'][0], # DEFAULT: 0.3
                                                              space_boundary['learning_rate'][1], num=SAMPLE_SIZE))), 
                'gamma': grid_search(list(np.linspace(space_boundary['gamma'][0], space_boundary['gamma'][1],
                                                     num=2, dtype=int))) # DEFAULT: 0
             }
        elif search_name == 'RandomSearch':
            hyperparams = { #TODO
                'max_depth': int(np.rint(np.random.uniform(space_boundary['max_depth'][0], space_boundary['max_depth'][1]))), # DEFAULT: 6
                'learning_rate': np.random.uniform(space_boundary['learning_rate'][0], space_boundary['learning_rate'][1]), # DEFAULT: 0.3
                'gamma': np.random.choice([space_boundary['gamma'][0], space_boundary['gamma'][1]]), # DEFAULT: 0
            }
        
        elif search_name == 'None':
            hyperparams = {
                'max_depth':6,
                'learning_rate':0.3,
                'gamma':0
            }
        hyperparams.update(fix_params)
            
    elif ml_lib == 'Sklearn':
        fix_params = {
            'loss': 'deviance',
            'subsample': 1.0,
            'criterion': 'friedman_mse',
            'min_samples_leaf': 1,
            'min_weight_fraction_leaf': 0,
            'min_impurity_decrease': 0.0,
            'min_impurity_split': None,
            'random_state': None,
            'max_features': None,
            'max_leaf_nodes': None
        }
        if search_name == 'GridSearch':
            hyperparams = {
                'max_depth': grid_search(list(np.linspace(space_boundary['max_depth'][0], 
                                                          space_boundary['max_depth'][1], num=SAMPLE_SIZE, dtype=int))),
                'learning_rate': grid_search(list(np.linspace(space_boundary['learning_rate'][0], 
                                                              space_boundary['learning_rate'][1], num=SAMPLE_SIZE))),  
                'n_estimators': grid_search(list(np.linspace(space_boundary['n_estimators'][0], 
                                                              space_boundary['n_estimators'][1], num=SAMPLE_SIZE, dtype=int))), 
                'min_samples_split': grid_search(list(np.linspace(space_boundary['min_samples_split'][0], 
                                                          space_boundary['min_samples_split'][1], num=SAMPLE_SIZE, dtype=int)))
            }
        elif search_name == 'RandomSearch':
            hyperparams = {
                'max_depth': int(np.rint(np.random.uniform(space_boundary['max_depth'][0], space_boundary['max_depth'][1]))),
                'learning_rate': np.random.uniform(space_boundary['learning_rate'][0], space_boundary['learning_rate'][1]),
                'n_estimators': int(np.rint(np.random.uniform(space_boundary['n_estimators'][0], 
                                                              space_boundary['n_estimators'][1]))),
                'min_samples_split': int(np.rint(np.random.uniform(space_boundary['min_samples_split'][0], 
                                                                   space_boundary['min_samples_split'][1])))
            }
        elif search_name == 'None':
            hyperparams = {
                'max_depth':3,
                'learning_rate':0.1,
                'n_estimators':100,
                'min_samples_split': 2
            }
        hyperparams.update(fix_params)
    
    config['hyperparams'] = hyperparams
    
    return config, fix_params


def get_space(search_name, ml_lib, space_boundary):
    # Get hp space for HyperOptSearch for specific ml library
    if ml_lib == 'XGBoost':
        return {'max_depth': hp.randint("max_depth",                                         space_boundary['max_depth'][0], space_boundary['max_depth'][1]),
                'learning_rate': hp.uniform('learning_rate', \
                                            space_boundary['learning_rate'][0], space_boundary['learning_rate'][1]),
                'gamma': hp.choice('gamma', \
                                   (space_boundary['gamma'][0], space_boundary['gamma'][1])), 
               }
    
    return {'max_depth': hp.randint("max_depth",                                         space_boundary['max_depth'][0], space_boundary['max_depth'][1]),
            'learning_rate': hp.uniform('learning_rate', \
                                        space_boundary['learning_rate'][0], space_boundary['learning_rate'][1]),
           'n_estimators': hp.randint('n_estimators', space_boundary['n_estimators'][0], space_boundary['n_estimators'][1]),
           'min_samples_split': hp.randint('min_samples_split', space_boundary['min_samples_split'][0], 
                                           space_boundary['min_samples_split'][1])
           }
        
    
def get_search_alg(search_name, ml_lib, space_boundary):
    # Get search algorithm object for HyperOptSearch
    if search_name == 'HyperOptSearch':
        return HyperOptSearch(space=get_space(search_name, ml_lib, space_boundary), metric='roc_auc', mode='max')

    return None


def get_space_boundary(ml_lib, search_alg):
    if search_alg == 'None': return None
    # Boundary for hp space, compatible with all search algs
    if ml_lib == 'XGBoost':
        return {'max_depth': [1, 6], # DEFAULT: 6
                'learning_rate': [0.1, 0.6], # DEFAULT: 0.3
                'gamma': [0, 1] # DEFAULT: 0
               }

    return {'max_depth': [1, 6], # DEFAULT: 3
            'learning_rate': [0.01, 0.5], # DEFAULT: 0.1
            'n_estimators': [50, 150], # DEFAULT: 100
            'min_samples_split': [2, 4] # DEFAULT: 2
           }


def trial_name_generator(trial):
    
    return trial.trainable_name + trial.trial_id


def perform_analysis(x_train, y_train, search_alg, ml_lib, scheduler):
     # Store data in object store to avoid copying large datasets mulitple times
    x_id = ray.put(x_train)
    y_id = ray.put(y_train) 
    
    time_start = time.strftime("%Y-%m-%d-%H;%M;%S")
    
    # Experiment setup
    trial_name = f'RayTune_{ml_lib}_{search_alg}_{time_start}'
    local_dir = 'C:/Users/edinmsa/Documents/hp_lib_results/ray_tune' 
    full_dir = local_dir + '/' + trial_name
    num_samples = 1 if search_alg == 'GridSearch' else NUM_ITERATIONS 
    stop_crit = {'training_iteration': 1} 
    scheduler = scheduler
    
    # Get hp space boundary for this specific ml library 
    space_boundary = get_space_boundary(ml_lib, search_alg)
    
    # Get hp space and search method obj for this specific ml library and search method
    search_algo = get_search_alg(search_alg, ml_lib, space_boundary) # This config is dict of list (all combinations of hps)
    config, fix_params = get_config(search_alg, ml_lib, x_id, y_id, space_boundary) # get ml_lib and search algo-specific experiment configuration
    
    # Add scheduler info to later statistics
    search_settings = dict()
    search_settings['scheduler'] = type(scheduler).__name__
    search_settings['space_samples'] = num_samples
    
    experiment_spec = Experiment(
                    name=trial_name, 
                    run=TrainableModel,
                    local_dir=local_dir,
                    resources_per_trial={'cpu': NUM_CPU, 'gpu': 0},
                    stop=stop_crit,
                    num_samples=num_samples,
                    trial_name_creator=trial_name_generator,
                    config=config
    )
    
    print(f'Performing hp tuning with {search_alg} on {ml_lib} models.')
    
    return run(experiment_spec, scheduler=scheduler, search_alg=search_algo, queue_trials=True), config,                         fix_params, space_boundary, search_settings, full_dir


def find_feat_types(dataset):
    ## Preparation for labelencoding
    feat_types = ['Categorical' if feature.name == 'category' else 'Numerical' for feature in dataset['data'].dtypes]
    
    cat_column = []
    num_column = []
    for index in range(len(feat_types)):
        if feat_types[index] == 'Categorical':
            cat_column.append(dataset['feature_names'][index])
        else:
            num_column.append(dataset['feature_names'][index])
            
    return num_column, cat_column


def run_pipeline(data_id, search_alg, ml_lib, scheduler):
    start = time.time()
    
    # Import data
    print(f"Loading data with id {data_id}...")
    dataset = fetch_openml(data_id=data_id, as_frame=True)

    print(dataset.DESCR[:445])
    print(f"Dataset '{dataset['details']['name']}' imported")

    x, y = dataset['data'], dataset['target']

    # Find feature types for preprocessing
    num_column, cat_column = find_feat_types(dataset) 

    # Convert categorical columns and target labels to int and normalize numerical data 
    for col in cat_column:
        x.loc[:, col] = LabelEncoder().fit_transform(x[col])
    if data_id == 1461: # Numerical data in dataset with data_id==151 is already normalized
        for col in num_column:
            x.loc[:, col] = StandardScaler().fit_transform(pd.DataFrame(x[col]))
    y = LabelEncoder().fit_transform(y)

    random_state = random.randint(0, 10000)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=random_state)
    print(f"Training set: {len(x_train)} samples,")
    print(f"Validation set: {len(y_test)} samples,") 
    print(f"Val/total ratio: {len(x_test)/len(y)*100:.1f}%")  
    

    analysis, config, fix_params, search_space_endpoints, search_settings, experiment_dir = perform_analysis(x_train,                                                                                                         y_train, search_alg, 
                                                                                                        ml_lib, scheduler)
    
    # Choose eval metric.
    metric = 'roc_auc' 
    
    best_config = analysis.get_best_config(metric, mode='max')
    
    print(f"Best config by {metric} is \n{best_config['hyperparams']}")
    
    # Train best model
    clf_best = get_clf(ml_lib, best_config['hyperparams']).fit(x_train, y_train)
    
    y_pred = clf_best.predict(x_test)
    
    # Evaluate best model on test set
    stats = statistics(clf_best, dataset, ml_lib, y_train, y_test, y_pred, metric, search_alg,                         config, search_space_endpoints, search_settings, fix_params, experiment_dir)

    print(f"Summary of best model on validation set: \n{stats}")
    
    # Export stats to .csv
    local_dir = 'C:/Users/edinmsa/Documents/hp_lib_results/ray_tune/statistics_csv/default/'
    
    time_stop = time.strftime("%Y-%m-%d-%H;%M;%S")
    file_name = f"hpsearch_raytune_ {ml_lib}_{search_alg}_{dataset['details']['name']}_{time_stop}.csv"
    file_dir = Path(local_dir)
    file_dir.mkdir(exist_ok=True)

    pd.DataFrame(stats).to_csv(file_dir / file_name)
    
    print(f"Exporting results to {local_dir + '/' + file_name}...")
    print("Export complete.")
    print('#'*80)
    
    return analysis.get_best_logdir(metric)



ray.init(include_dashboard=True, ignore_reinit_error=True)
    
# Settings
SAMPLE_SIZE = 1 # Number of samples of hps in the hp search space
NUM_ITERATIONS = 1 # for all searches except GridSearch (which has SAMPLE_SIZE**dim_search_space)
NUM_CPU = 6

if __name__ == '__main__':
    
    
    # Iterate over datasets, ml libs, search strategy
    data_ids = [1461, 151]
    search_algs = ['None', 'GridSearch', 'RandomSearch', 'HyperOptSearch']
    ml_libs = ['XGBoost', 'Sklearn']
    schedulers = [FIFOScheduler(), ASHAScheduler()]
    
    for data_id in data_ids:
        for search_alg in search_algs:
            for ml_lib in ml_libs: 
                for scheduler in schedulers:
                    logdir = run_pipeline(data_id, search_alg, ml_lib, scheduler)
                
                print('Run')
                print(f'$ tensorboard --logdir={logdir}')
                print('to see stats of best model in Tensorboard')


# In[ ]:




