import numpy as np
from sklearn.model_selection import KFold
import os
import re
from os import listdir
from os.path import isfile, join
from subprocess import PIPE, run
import pandas as pd
import glob
import networkx as nx
from networkx.drawing.nx_pydot import write_dot



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
    cmd = f"{bin_path} -i {matrix_file} -n {n} -n2 {n2} -m {m} -t {tree_path} -r 1 -l 0 -fd 0.0001 -ad 0.0001 -cc 0.0 -s  -o {output_prefix} -seed {seed}"
    result = run(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    #print(cmd)

    out = result.stdout
    out = out.split('\n')
    score = -1
    for line in out:
        if "True tree score:" in line:
            score = np.mean([float(x) for x in line.split('\t')[1:]])
            break

    while score == -1:
        result = run(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
        out = result.stdout
        out = out.split('\n')
        for line in out:
            if "True tree score:" in line:
                score = np.mean([float(x) for x in line.split('\t')[1:]])
                break

    # if score == -1:
    #     print(cmd)
    #     raise RuntimeError(
    #     "Error: mtSCITE did not run successfully or its output is missing the line 'True tree score'. "
    #     "Please check the command output and ensure the binary is functioning as expected. "
    #     f"Command output:\n{result.stdout}\nCommand error:\n{result.stderr}"
    # )
    
    # Remove intermediate files from this step
    #os.remove(matrix_file)
    #os.remove(f'{output_prefix}.samples')

    return float(score)

def learn_mtscite(mat, output_dir, suffix="temp", bin_path="./mtscite", l=200000, seed=1, **kwargs):
    n, m = mat.shape
    print("Shape of the mutation matrix used for error rate learning")
    print(n,m)
    # Write mat
    matrix_file = os.path.join(output_dir, f'mat{suffix}.txt')
    np.savetxt(matrix_file, mat, delimiter=' ')

    output_prefix = os.path.join(output_dir, f'learned{suffix}') 
    
    cmd = f"{bin_path} -i {matrix_file} -n {n} -m {m} -r 1 -l {l} -max_treelist_size 100 -fd 0.0001 -ad 0.0001 -cc 0.0 -s -a -o {output_prefix} -seed {seed}"
    result = run(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    print(cmd)
    print(result)

    os.remove(matrix_file)
    out = result.stdout
    out = out.split('\n')
    score = -1

    for line in out:
        if "best log score for tree:" in line:
            score = line.split('\t')[1]
        

    if score == -1:
        raise RuntimeError(
        "Error: mtSCITE did not run successfully or its output is missing the line 'True tree score'. "
        "Please check the command output and ensure the binary is functioning as expected. "
        f"Command output:\n{result.stdout}\nCommand error:\n{result.stderr}"
    )

    # remove unnecessary files and only return first map tree
    #print(f'{output_prefix}_map0.newick')
    #os.remove(f'{output_prefix}.samples')
    #os.remove(f'{output_prefix}_map0.newick')
    return f"{output_prefix}_map0.gv", float(score)

def generate_star_tree(num_mutations):

    star_tree = nx.DiGraph()
    center_node = num_mutations + 1
    star_tree.add_edges_from((center_node, i) for i in range(1, num_mutations + 1))

    return star_tree

def kfold_mtscite(data_path, k=3, rate=0., seed=42, **kwargs):

    X = np.loadtxt(data_path, delimiter=' ')

    # Filter the matrix for learning: keep muts with at least 2 cells with > .99
    filtered_X = X[np.sum(X > .90, axis=1) >= 2, :]
    removed_X = X[np.sum(X > .90, axis=1) < 2, :]
    print(f"Full size: {X.shape}")
    print(f"Filtered size: {filtered_X.shape}")
    full_X = np.vstack([filtered_X, removed_X])

    num_mutations = X.shape[0]
    num_mutations_filtered = filtered_X.shape[0]
    
    if num_mutations_filtered < 10: # don't learn
        return np.nan, np.nan

    # generate star tree to normalise against
    star_tree = generate_star_tree(num_mutations_filtered)
    # TODO potentially make name with error rate!
    star_tree_file_path = os.path.join(output_dir, "star_tree.gv")
    write_dot(star_tree, star_tree_file_path)
    
    val_scores = []
    val_star = []
    
    if k == 1:
        tree_path, val_ll = learn_mtscite(X, output_dir=output_dir, **kwargs)
        val_ll = score_tree(X, tree_path, suffix=f'_{rate}_{seed}_{k}', **kwargs)
        val_ll_star_tree = score_tree(X, star_tree_file_path, suffix=f'_{rate}', **kwargs)
        
        val_scores.append(val_ll)
        val_star.append(val_ll_star_tree)
        return val_scores, val_star

    kf = KFold(n_splits=k, random_state=seed, shuffle=True)
    for i, (train_index, test_index) in enumerate(kf.split(full_X.T)):
        
        # Learn on filtered training data
        train_data = filtered_X.T[train_index].T
        

        # suffix is the error rate, the repetition number and the fold number
        suffix = f"_{rate}_{seed}_{i}"
        tree_path, train_ll = learn_mtscite(train_data, suffix=suffix, **kwargs)
        print(f"tree path {tree_path}")

        # Evaluate on complete test data
        test_data = full_X.T[test_index].T

        #print("Learned tree")
        # suffix before was f"_{rate}"
        val_ll = score_tree(test_data, tree_path, filtered_X.shape[0], suffix=suffix, **kwargs)
        #print("Star tree")
        val_ll_star_tree = score_tree(test_data, star_tree_file_path, filtered_X.shape[0], suffix=suffix, **kwargs)

        log_lik_normalised = val_ll / val_ll_star_tree
        print(f"test tree score: {val_ll}")
        print(f"star tree score: {val_ll_star_tree}")
        
        #print(f"reported score {diff_log_lik}")
        # Store ll on complete test data normalised by score on star tree
        val_scores.append(log_lik_normalised)
        val_star.append(val_ll_star_tree)
        

    return val_scores, val_star


def score_error_rates(data_path, **kwargs):
    scores = dict()
    stars = dict()
    
    mats = [join(data_path, f) for f in listdir(data_path) if isfile(join(data_path, f))]
    #print(mats)
    for mat in mats:
        # only use basename
        mat_filename = os.path.basename(mat)
        # Check if the filename ends with '.csv' and starts with a numeric pattern
        if mat_filename.endswith('.csv') and mat_filename.startswith('0.'): #re.match(r'^\d+(\.\d*)?\.csv$', mat):
            print(mat)
            error_rate = mat.split("/")[-1].split(".csv")[0] # read error rate from input file
            print("\n")
            print(f'Learning the tree for error rate {error_rate}')
            out1, out2 = kfold_mtscite(mat, rate=error_rate, **kwargs) # run CV for this error rate 
            if out1 is not None:
                scores[error_rate] = out1

            if out2 is not None:
                stars[error_rate] = out2

                
    return scores, stars



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
    df2_list = []
    
    for rep in range(r):
        val_scores, stars = score_error_rates(data_directory, bin_path=mtscite_bin, l=l, k=k, seed=rep, output_dir=output_dir)
        print(val_scores)
        df = pd.DataFrame(val_scores)
        df_list.append(df)

        df2 = pd.DataFrame(stars)
        df2_list.append(df2)


    full_df = pd.concat(df_list)
    #print(full_df)
    full_df.to_csv(os.path.join(output_dir, 'val_scores_normalised.txt'))

    full_df_stars = pd.concat(df2_list)
    full_df_stars.to_csv(os.path.join(output_dir, 'val_scores_star_trees.txt'))
