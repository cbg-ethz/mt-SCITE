import numpy as np
from sklearn.model_selection import KFold
import os
from os import listdir
from os.path import isfile, join
from subprocess import PIPE, run
import pandas as pd
import glob




def score_tree(mat, tree_path, n2, bin_path="./mtscite", seed=1, suffix="temp", **kwargs):
    """
    n2 is the number of mutations in the true tree. The remaining mutations will be set as non-mutated in the LL computation
    """

    matrix_file = os.path.join(output_dir, f'mat{suffix}.txt')
    np.savetxt(matrix_file, mat, delimiter=' ')

    output_prefix = os.path.join(output_dir, f'output{suffix}') 
    #np.savetxt(f'mat{suffix}.txt', mat, delimiter=' ')
    
    n, m = mat.shape
    print("Shape of the mutation matrix used for error rate testing")
    print(n, m)
    cmd = f"{bin_path} -i {matrix_file} -n {n} -n2 {n2} -m {m} -t {tree_path} -r 1 -l 0 -fd 0.0001 -ad 0.0001 -cc 0.0 -s -a -o {output_prefix} -seed {seed}"
    result = run(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)

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
    
    # Remove intermediate files from this step
    os.remove(matrix_file)
    os.remove(f'{output_prefix}.samples')

    return float(score)

def learn_mtscite(mat, output_dir, suffix="temp", bin_path="./mtscite", l=200000, seed=1, **kwargs):
    n, m = mat.shape
    print("Shape of the mutation matrix used for error rate learning")
    print(n,m)
    # Write mat
    matrix_file = os.path.join(output_dir, f'mat{suffix}.txt')
    np.savetxt(matrix_file, mat, delimiter=' ')

    output_prefix = os.path.join(output_dir, f'learned{suffix}') 
    
    cmd = f"{bin_path} -i {matrix_file} -n {n} -m {m} -r 1 -l {l} -max_treelist_size 1 -fd 0.0001 -ad 0.0001 -cc 0.0 -s -o {output_prefix} -seed {seed}"
    result = run(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    os.remove(matrix_file)
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

    # remove unnecessary files and only return first map tree
    #print(f'{output_prefix}_map0.newick')
    os.remove(f'{output_prefix}.samples')
    os.remove(f'{output_prefix}_map0.newick')
    return f"{output_prefix}_map0.gv", float(score)

def kfold_mtscite(data_path, k=3, rate=0., seed=42, **kwargs):
    X = np.loadtxt(data_path, delimiter=' ')

    if X.shape[0] < 10: # don't learn
        return None
    
    val_scores = []
    if k == 1:
        tree_path, val_ll = learn_mtscite(X, output_dir=output_dir, **kwargs)
        val_ll = score_tree(X, tree_path, suffix=f'_{rate}', **kwargs)
        val_ll = val_ll / X.size
        val_scores.append(val_ll)
        return val_scores

    kf = KFold(n_splits=k, random_state=seed, shuffle=True)
    for i, (train_index, test_index) in enumerate(kf.split(X.T)):
        # Learn on filtered training data
        train_data = X.T[train_index].T
        #tree_path, train_ll = learn_mtscite(train_data, suffix=f"_{rate}_{i}", **kwargs)
        tree_path, train_ll = learn_mtscite(train_data, suffix=f"_{rate}_{i}", **kwargs)
        # Evaluate on complete test data
        test_data = X.T[test_index].T
        val_ll = score_tree(test_data, tree_path, X.shape[0], suffix=f"_{rate}", **kwargs)
        # Store ll on complete test data divided by its size
        val_scores.append(val_ll/test_data.size)

    return val_scores


def score_error_rates(data_path, **kwargs):
    scores = dict()
    mats = [join(data_path, f) for f in listdir(data_path) if isfile(join(data_path, f))]
    for mat in mats:
        if 'csv' in mat:
            error_rate = mat.split("/")[-1].split(".csv")[0] # read error rate from input file
            out = kfold_mtscite(mat, rate=error_rate, **kwargs) # run CV for this error rate
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
        val_scores = score_error_rates(data_directory, bin_path=mtscite_bin, l=l, k=k, seed=rep, output_dir=output_dir)
        df = pd.DataFrame(val_scores)
        df_list.append(df)
    full_df = pd.concat(df_list)
    full_df.to_csv(os.path.join(output_dir, 'val_scores.txt'))
