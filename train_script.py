import numpy as np
import pandas as pd
import pickle
import itertools
import sklearn
from sklearn.linear_model import LogisticRegression
import os
import time
import argparse
import utils 
import importlib

def calc_frontier(method, params_combs, metrics, log=True):
    frontier = [] 
    for i,params in enumerate(utils.product_dicts(params_combs)):
        time1=time.time()
        learner = method(**params)
        learner.train(tr_X, tr_Xp, tr_y)
        train_time = time.time() - time1
        
        time1=time.time()
        train_preds = learner.predict(tr_X, tr_Xp)
        preds = learner.predict(test_X, test_Xp)
        predict_time = time.time()-time1

        point_dict = {'method': learner.name, 'params': params}
        metric_dict = {'train_time': train_time, 'predict_time': predict_time}
        for metric_name, metric in exp_metrics:
            metric_dict['train_'+metric_name] = metric(train_preds, tr_X, tr_Xp, tr_y)
            metric_dict[metric_name] = metric(preds, test_X, test_Xp, test_y)
        point_dict['metrics'] = metric_dict 
        if log:
            print("=======================================================================================================")
            print(f"{i}th param set for {learner.name} done, train time: {train_time:.5f}, predict_time: {predict_time:.5f}")
            for metric_name, metric in exp_metrics:
                print(f"train_{metric_name}: {metric_dict['train_'+metric_name]:.7f}, {metric_name}: {metric_dict[metric_name]:.7f}")
        frontier.append(point_dict)
    return frontier

def average_dicts(dicts):
    avg_dict = {}
    keys = dicts[0].keys()
    n = len(dicts)
    for key in keys:
        avg_dict[key] = np.mean([dict_[key] for dict_ in dicts])
    return avg_dict

def average_frontiers(frontiers_list):
    avg_frontiers = []
    matched_frontiers = zip(*frontiers_list)
    for matched_frontier_list in matched_frontiers: 
        frontier = []
        matched_points = zip(*matched_frontier_list)
        for matched_point in matched_points:
            expd = matched_point[0] 
            point_dict = {'method': expd['method'], 'params': expd['params']}
            point_dict['metrics'] = average_dicts([pd['metrics'] for pd in matched_point])
            frontier.append(point_dict)
        avg_frontiers.append(frontier)
    return avg_frontiers


#example usage: python3 train_script.py --expfile ind_exp --dataset german --expname german_ind --ntrials 10
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--expfile', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--expname', type=str)
    parser.add_argument('--ntrials', type=int)
    args = parser.parse_args()

    expfile = importlib.import_module(f"experiments.{args.expfile}")
    with open(f'dataset/{args.dataset}.pkl', 'rb') as f:
        X, Xp, y, proatts = pickle.load(f)
    y = y.astype(int)
    # at the end code averaging everything up boi
    # add method.name
    # add metrics to expfile

    total_frontiers = []
    for trial in range(args.ntrials):
        (tr_X, tr_Xp, tr_y), (test_X, test_Xp, test_y) = utils.tr_test_split(X, Xp, y)
        frontiers = []
        for method, params_combs, exp_metrics in zip(expfile.methods, expfile.params_list, expfile.metrics_list):
            frontiers.append(calc_frontier(method, params_combs, exp_metrics))
        total_frontiers.append(frontiers)

    average_frontiers = average_frontiers(total_frontiers)


    if not os.path.exists('results'):
        os.makedirs('results')

    with open(f'results/{args.expname}.pkl', 'wb') as f:
        pickle.dump(average_frontiers, f)

