import numpy as np
from sklearn.model_selection import KFold
import os
from os import listdir
from os.path import isfile, join
from subprocess import PIPE, run
import matplotlib.pyplot as plt
import pandas as pd

def score_tree(mat, tree_path, bin_path="./mtscite", seed=1, suffix="temp", **kwargs):
    np.savetxt(f'mat_{suffix}.txt', mat, delimiter=' ')
    n, m = mat.shape
    cmd = f"{bin_path} -i mat_{suffix}.txt -n {n} -m {m} -t {tree_path} -r 1 -l 1 -s -a -o output -seed {seed}"
    result = run(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    os.remove(f'mat_{suffix}.txt')
    out = result.stdout
    out = out.split('\n')
    score = -1
    for line in out:
        if "best log score for tree:" in line:
            score = line.split('\t')[1]
            break
    return float(score)

def learn_mtscite(mat, bin_path="./mtscite", l=200000, seed=1, suffix="temp", **kwargs):
    n, m = mat.shape
    # Write mat
    np.savetxt(f'mat_{suffix}.txt', mat, delimiter=' ')
    cmd = f"{bin_path} -i mat_{suffix}.txt -n {n} -m {m} -r 1 -l {l} -s -a -o learned -seed {seed}"
    result = run(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    os.remove(f'mat_{suffix}.txt')
    out = result.stdout
    out = out.split('\n')
    score = -1
    for line in out:
        if "best log score for tree:" in line:
            score = line.split('\t')[1]
            break
    return "learned_map0.gv"

def kfold_mtscite(data_path, k=10, **kwargs):
    print(data_path)
    X = np.loadtxt(data_path, delimiter=' ')
    kf = KFold(n_splits=k)
    val_scores = []
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        train_data = X[train_index]
        test_data = X[test_index]
        tree_path = learn_mtscite(train_data, **kwargs)
        val_ll = score_tree(test_data, tree_path, **kwargs)
        val_scores.append(val_ll)
    n_nonzero = np.count_nonzero(X)
    val_scores = np.array(val_scores) / n_nonzero
    return val_scores

def score_error_rates(data_path, **kwargs):
    scores = dict()
    mats = [join(data_path, f) for f in listdir(data_path) if isfile(join(data_path, f))]
    for mat in mats:
        if 'csv' in mat:
            error_rate = mat.split("/")[-1].split(".csv")[0]
            scores[error_rate] = kfold_mtscite(mat, **kwargs)
    return scores


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('directory')
parser.add_argument('mtscite_bin_path')
parser.add_argument('-l', default=200000)
parser.add_argument('-k', default=3)
args = parser.parse_args()

if __name__ == "__main__":
    data_directory = args.directory
    mtscite_bin = args.mtscite_bin_path
    l = args.l
    k = args.k
    val_scores = score_error_rates(data_directory, bin_path=mtscite_bin, l=l, k=k)
    df = pd.DataFrame(val_scores)
    df.to_csv('val_scores.csv')

