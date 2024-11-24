import numpy as np
from sklearn.model_selection import KFold
import os
from os import listdir
from os.path import isfile, join
from subprocess import PIPE, run
import pandas as pd




def score_tree(mat, tree_path, n2, bin_path="./mtscite", seed=1, suffix="temp", **kwargs):
    """
    n2 is the number of mutations in the true tree. The remaining mutations will be set as non-mutated in the LL computation
    """
    np.savetxt(f'mat{suffix}.txt', mat, delimiter=' ')
    n, m = mat.shape
    print(n, m)
    cmd = f"{bin_path} -i mat{suffix}.txt -n {n} -n2 {n2} -m {m} -t {tree_path} -r 1 -l 0 -fd 0.0001 -ad 0.0001 -cc 0.0 -s -a -o output{suffix} -seed {seed}"
    result = run(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    os.remove(f'mat{suffix}.txt')
    out = result.stdout
    out = out.split('\n')
    score = -1
    for line in out:
        if "True tree score:" in line:
            score = np.mean([float(x) for x in line.split('\t')[1:]])
            break
    if score == -1:
        raise RuntimeError(
        "Error: mtSCITE did not run successfully or its output is missing the line 'True tree score'. "
        "Please check the command output and ensure the binary is functioning as expected. "
        f"Command output:\n{result.stdout}\nCommand error:\n{result.stderr}"
    )
        
    return float(score)

def learn_mtscite(mat, bin_path="./mtscite", l=200000, seed=1, suffix="temp", **kwargs):
    n, m = mat.shape
    print(n,m)
    # Write mat
    np.savetxt(f'mat{suffix}.txt', mat, delimiter=' ')
    cmd = f"{bin_path} -i mat{suffix}.txt -n {n} -m {m} -r 1 -l {l}  -fd 0.0001 -ad 0.0001 -cc 0.0 -s -o learned{suffix} -seed {seed}"
    result = run(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    os.remove(f'mat{suffix}.txt')
    out = result.stdout
    out = out.split('\n')
    score = -1
    for line in out:
        if "best log score for tree:" in line:
            score = line.split('\t')[1]
            break
    if score == -1:
        raise RuntimeError(
        "Error: mtSCITE did not run successfully or its output is missing the line 'True tree score'. "
        "Please check the command output and ensure the binary is functioning as expected. "
        f"Command output:\n{result.stdout}\nCommand error:\n{result.stderr}"
    )
    
    return f"learned{suffix}_map0.gv", float(score)

def kfold_mtscite(data_path, k=3, rate=0., seed=42, **kwargs):
    print(data_path)
    X = np.loadtxt(data_path, delimiter=' ')

    # Filter the matrix for learning: keep muts with at least 2 cells with > .99
    filtered_X = X[np.sum(X > .99, axis=1) >= 2, :]
    removed_X = X[np.sum(X > .99, axis=1) < 2, :]
    print(f"Full size: {X.shape}")
    print(f"Filtered size: {filtered_X.shape}")
    full_X = np.vstack([filtered_X, removed_X])

    if filtered_X.shape[0] <= 10: # don't learn
        return None

    val_scores = []
    if k == 1:
        tree_path, val_ll = learn_mtscite(X, **kwargs)
        val_ll = score_tree(X, tree_path, suffix=f"_{rate}", **kwargs)
        val_ll = val_ll / X.size
        val_scores.append(val_ll)
        return val_scores

    kf = KFold(n_splits=k, random_state=seed, shuffle=True)
    for i, (train_index, test_index) in enumerate(kf.split(full_X.T)):
        # Learn on filtered training data
        train_data = filtered_X.T[train_index].T
        tree_path, train_ll = learn_mtscite(train_data, suffix=f"_{rate}_{i}", **kwargs)
        print(f"Merged size: {full_X.shape}")
        # Evaluate on complete test data
        test_data = full_X.T[test_index].T
        val_ll = score_tree(test_data, tree_path, filtered_X.shape[0], suffix=f"_{rate}", **kwargs)
        #print(f"Val log likelihood {val_ll}")
        # Store ll on complete test data divided by its size
        normalised_val_ll = val_ll / test_data.size
        #print(f"Val normalised log likelihood {normalised_val_ll}")
        val_scores.append(normalised_val_ll)

        
    return val_scores


def score_error_rates(data_path, **kwargs):
    scores = dict()
    mats = [join(data_path, f) for f in listdir(data_path) if isfile(join(data_path, f))]
    for mat in mats:
        if 'csv' in mat:
            error_rate = mat.split("/")[-1].split(".csv")[0] # read error rate from input file
            print(f"Performing k-fold cross validation for error rate {error_rate}")
            out = kfold_mtscite(mat, **kwargs) # run CV for this error rate
            if out is not None:
                scores[error_rate] = out
    return scores



import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--directory', default='/cluster/work/bewi/members/pedrof/mtscite/YFV2001_matrix_output_final/YFV2001_matrix_output')
parser.add_argument('--mtscite_bin_path', default='/cluster/work/bewi/members/pedrof/mtscite/mt-SCITE-main/mtscite')
parser.add_argument('-l', default=200000, type=int, help="Number of MCMC iterations")
parser.add_argument('-k', default=3, type=int, help="Number of CV folds")
parser.add_argument('-r', default=10, type=int, help="Number of repetitions of the CV scheme for each error rate")
parser.add_argument('-o', default=".", help="Output directory")

args = parser.parse_args()

if __name__ == "__main__":
    data_directory = args.directory
    mtscite_bin = args.mtscite_bin_path
    l = args.l
    k = args.k
    r = args.r
    output_dir = args.o

    df_list = []
    for rep in range(r):
        val_scores = score_error_rates(data_directory, bin_path=mtscite_bin, l=l, k=k, seed=rep)
        df = pd.DataFrame(val_scores)
        df_list.append(df)
    full_df = pd.concat(df_list)
    full_df.to_csv(os.path.join(output_dir, 'val_scores.csv'))
