# import libs
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
    Compute the log-likelihood score of the training tree on the heldout test data.

    This function:
    1. Saves the test set mutation matrix to a temporary text file.
    2. Calls the mtSCITE binary to compute log-likelihood scores for the training tree
       and the mutation matrix.
    3. Parses mtSCITE’s output, extracts the line containing 'True tree score:', 
       and returns that value as a float.
    4. Removes temporary files before returning.

    Parameters
    ----------
    mat : np.ndarray
        Mutation matrix (test set) to be scored. Rows represent mutations, columns samples.
    tree_path : str
        Path to the tree (in .gv or .newick) whose score we want to compute.
    n2 : int
        Number of mutations in the training tree. The remaining will be treated as non-mutated.
    output_dir : str
        Directory where temporary files and output files are written.
    bin_path : str, optional
        Path to the mtSCITE binary.
    seed : int, optional
        Random seed for reproducibility, by default 1.
    suffix : str, optional
        Suffix to append to output filenames, by default 'temp'.

    Returns
    -------
    float
        Computed log-likelihood score for the provided tree.

    Raises
    ------
    RuntimeError
        If the mtSCITE output does not contain 'True tree score' or if the call fails.
    """

    # Save the mutation matrix to file such that it can be used by mt-SCITE
    matrix_file = os.path.join(output_dir, f'mat{suffix}.txt')
    np.savetxt(matrix_file, mat, delimiter=' ')

    output_prefix = os.path.join(output_dir, f'output{suffix}')

    n, m = mat.shape
    print("Shape of the mutation matrix used for error rate testing")
    print(n, m)

    # Build mt-SCITE command
    cmd = f"{bin_path} -i {matrix_file} -n {n} -n2 {n2} -m {m} -t {tree_path} -r 1 -l 0 -fd 0.0001 -ad 0.0001 -cc 0.0 -s -a -o {output_prefix} -seed {seed}"
    result = run(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)

    # Parse mtSCITE's stdout for the tree score
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

    # Remove intermediate files
    if os.path.exists(matrix_file):
        os.remove(matrix_file)
    samples_file = f'{output_prefix}.samples'
    if os.path.exists(samples_file):
        os.remove(samples_file)

    return float(score)

def learn_mtscite(mat, output_dir, suffix="temp", bin_path="./mtscite", l=200000, seed=1, **kwargs):

    """
    Run mtSCITE to learn an optimal tree from the mutation matrix (training set).

    This function:
    1. Saves the input mutation matrix to a temporary file.
    2. Calls the mtSCITE binary to learn a tree (runs MCMC).
    3. Parses mtSCITE's output for the best log score.
    4. Removes unnecessary mtSCITE output files and returns the path to the optimal tree and the score.

    Parameters
    ----------
    mat : np.ndarray
        Mutation matrix. Rows typically represent mutations, columns samples.
    output_dir : str
        Directory where temporary files and results will be written.
    suffix : str, optional
        Suffix appended to output filenames, by default 'temp'.
    bin_path : str, optional
        Path to the mtSCITE binary, by default './mtscite'.
    l : int, optional
        Number of MCMC iterations, by default 200000.
    seed : int, optional
        Random seed, by default 1.

    Returns
    -------
    Tuple[str, float]
        Path to the learned tree in .gv format, and the best log score from mtSCITE.

    Raises
    ------
    RuntimeError
        If the mtSCITE output is missing the line needed for identifying the best log score.
    """

    n, m = mat.shape
    print("Shape of the mutation matrix used for error rate learning")
    print(n,m)

    # Write the matrix to a temporary file
    matrix_file = os.path.join(output_dir, f'mat{suffix}.txt')
    np.savetxt(matrix_file, mat, delimiter=' ')

    output_prefix = os.path.join(output_dir, f'learned{suffix}')

    # Build and run mtSCITE command
    cmd = f"{bin_path} -i {matrix_file} -n {n} -m {m} -r 1 -l {l} -max_treelist_size 1 -fd 0.0001 -ad 0.0001 -cc 0.0 -s -o {output_prefix} -seed {seed}"
    result = run(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    
    os.remove(matrix_file)

    # Parse mtSCITE's stdout for the best log score
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


    return f"{output_prefix}_map0.gv", float(score)

def generate_star_tree(num_mutations):
    """
    Generate a 'star' shaped DiGraph with a single center node and edges to all other nodes.

    Parameters
    ----------
    num_mutations : int
        Number of mutations; the star tree will have 1 center node and `num_mutations` leaves.
    """


    star_tree = nx.DiGraph()
    center_node = num_mutations + 1
    star_tree.add_edges_from((center_node, i) for i in range(1, num_mutations + 1))

    return star_tree

def kfold_mtscite(data_path, k=3, rate=0., seed=42, **kwargs):
    """
    Perform k-fold cross-validation with mtSCITE on a mutation matrix.

    For k=1, it trains on the whole dataset and returns a single score.
    For k>1, it splits columns (cells) into train/test sets, learns a tree on the train portion, 
    and computes likelihood scores on the test portion.  Also computes each test score 
    with a 'star tree' baseline.
    """
    X = np.loadtxt(data_path, delimiter=' ')

    num_mutations = X.shape[0]

    if num_mutations < 7: # don't learn
        return np.nan, np.nan

    # Generate a star tree to compare as a baseline
    star_tree = generate_star_tree(num_mutations)
    star_tree_file_path = os.path.join(output_dir, "star_tree.gv")
    write_dot(star_tree, star_tree_file_path)

    if k == 1:
        # Train on the entire dataset
        tree_path, val_ll = learn_mtscite(X, output_dir=output_dir, **kwargs)

        # suffix is the error rate and the repetition number
        suffix = f"_{rate}_{seed}"
        
        val_ll = score_tree(X, tree_path, suffix=suffix, **kwargs)
        val_ll_star_tree = score_tree(X, star_tree_file_path, suffix=suffix, **kwargs)

        if normalised:
            val_ll = val_ll / val_ll_star_tree
        
        return [val_ll], [val_ll_star_tree]

    # Otherwise, perform k-fold cross-validation
    val_scores = []
    val_star = []

    kf = KFold(n_splits=k, random_state=seed, shuffle=True)
    for i, (train_index, test_index) in enumerate(kf.split(X.T)):
        # Learn on training data
        train_data = X.T[train_index].T

        # suffix is the error rate, the repetition number and the fold number
        suffix = f"_{rate}_{seed}_{i}"
        
        tree_path, train_ll = learn_mtscite(train_data, suffix=suffix, **kwargs)
        #print(f"tree path {tree_path}")

        # Evaluate on test data
        test_data = X.T[test_index].T
        val_ll = score_tree(test_data, tree_path, X.shape[0], suffix=suffix, **kwargs)
        val_ll_star_tree = score_tree(test_data, star_tree_file_path, X.shape[0], suffix=suffix, **kwargs)

        print(f"Fold {i} - Learned tree score: {val_ll}, Star tree score: {val_ll_star_tree}")

        if normalised:
            val_ll = val_ll / val_ll_star_tree

        val_scores.append(val_ll)
        val_star.append(val_ll_star_tree)

    return val_scores, val_star


def score_error_rates(data_path, **kwargs):
     """
    Traverse files in the specified data directory, identify those that start with '0.'
    and end with '.csv' (interpreted as error rates), then run kfold_mtscite for each file.

    Parameters
    ----------
    data_path : str
        Directory containing multiple .csv mutation probability matrix
        files, each  with a  prefix denoting the error rate.
    output_dir : str
        Directory where results and temporary files will be written.
    **kwargs :
        Additional arguments passed to kfold_mtscite.

    Returns
    -------
    Tuple[Dict[str, List[float]], Dict[str, List[float]]]
        A 2-tuple of dictionaries:
            - scores[error_rate] = list of log-likelihoods of the
              training tree on the test data
            - stars[error_rate]  = list of log-likelihoods of the
               star-tree on the test data
    """
     
     scores = dict()
     stars = dict()

     mats = [join(data_path, f) for f in listdir(data_path) if isfile(join(data_path, f))]
     #print(mats)
     for mat in mats:
        # only use basename
        mat_filename = os.path.basename(mat)
        # Check if the filename ends with '.csv' and starts with a numeric pattern
        if mat_filename.endswith('.csv') and mat_filename.startswith('0.'):
            #print(mat)
            # The portion before .csv is assumed to be the error rate
            error_rate = mat.split("/")[-1].split(".csv")[0] # read error rate from input file
            print(f"Running k-fold cross-validation for error rate: {error_rate}")
            out1, out2 = kfold_mtscite(mat, rate=error_rate, **kwargs)
            if out1 is not None:
                scores[error_rate] = out1

            if out2 is not None:
                stars[error_rate] = out2

     return scores, stars


import argparse

parser = argparse.ArgumentParser(
        description="Run mtSCITE with k-fold CV across multiple error rates."
    )
parser.add_argument('--directory', default='/cluster/work/bewi/members/pedrof/mtscite/YFV2001_matrix_output_final/YFV2001_matrix_output')
parser.add_argument('--mtscite_bin_path', default='/cluster/work/bewi/members/pedrof/mtscite/mt-SCITE-main/mtscite')
parser.add_argument('-l', default=200000, type=int, help="Number of MCMC iterations")
parser.add_argument('-k', default=3, type=int, help="Number of CV folds")
parser.add_argument('-r', default=3, type=int, help="Number of repetitions of the CV scheme for each error rate")
parser.add_argument('-o', default=".", help="Output directory")
parser.add_argument('-n', default="True", type=bool, help="Whether (T) or not (F) the tree scores should be normalised by the star tree score")

args = parser.parse_args()

if __name__ == "__main__":
    data_directory = args.directory
    mtscite_bin = args.mtscite_bin_path
    l = args.l
    k = args.k
    r = args.r
    output_dir = args.o
    normalised = args.n

    df_list = []
    df2_list = []

    # Collect results across multiple repetitions
    for rep in range(r):
        print(f"Running repetition {rep+1} out of {r} ") 
        val_scores, stars = score_error_rates(data_directory, bin_path=mtscite_bin, l=l, k=k, seed=rep, output_dir=output_dir, normalised=normalised)
        print(val_scores)
        df = pd.DataFrame(val_scores)
        df_list.append(df)

        df2 = pd.DataFrame(stars)
        df2_list.append(df2)

    full_df = pd.concat(df_list)
    print(full_df)
    full_df.to_csv(os.path.join(output_dir, 'val_scores.txt'))

    full_df_stars = pd.concat(df2_list)
    full_df_stars.to_csv(os.path.join(output_dir, 'val_scores_star_trees.txt'))
