import marimo

__generated_with = "0.8.3"
app = marimo.App()


@app.cell
def __():
    from collections import Counter
    import os
    from glob import glob
    import re
    import numbers
    import itertools

    from graphviz import Source
    from IPython.display import display, Markdown
    import matplotlib.pyplot as plt
    from matplotlib.ticker import EngFormatter
    import networkx as nx
    import numpy as np
    from scipy import stats
    import pandas as pd
    import seaborn as sns
    from matplotlib.ticker import FormatStrFormatter



    sns.set_style('white')
    sns.set_context('notebook')

    def plot_style(figsize=(12, 6), labelsize=20, titlesize=24, ticklabelsize=14, **kwargs):
       basic_style = {
           'figure.figsize': figsize,
           'axes.labelsize': labelsize,
           'axes.titlesize': titlesize,
           'xtick.labelsize': ticklabelsize,
           'ytick.labelsize': ticklabelsize,
           'axes.spines.top': False,
           'axes.spines.right': False,
           'axes.spines.left': False,
           'axes.grid': False,
           'axes.grid.axis': 'y',
       }
       basic_style.update(kwargs)
       return plt.rc_context(rc=basic_style)

    blue = sns.xkcd_rgb['ocean blue']
    return (
        Counter,
        EngFormatter,
        FormatStrFormatter,
        Markdown,
        Source,
        blue,
        display,
        glob,
        itertools,
        np,
        numbers,
        nx,
        os,
        pd,
        plot_style,
        plt,
        re,
        sns,
        stats,
    )


@app.cell
def __():
    # specify directories

    mtscite_output_dir = '../../../results/data_analysis/p9855/error_rate_learning/'
    return mtscite_output_dir,


@app.cell
def __(mo):
    mo.md(r"""## Create df with performance data""")
    return


@app.cell
def __(mo):
    mo.md(r"""### Clone map""")
    return


@app.cell
def __(pd):
    data = [['bulk', 's0', 'P3861_218.clean.dedup_ac.txt', 'ss0'], 
            ['E', 's1', 'P9855_2085_S108_L004_ac.txt', 'ss1'],
            ['A', 's2', 'P9855_2089_S112_L004_ac.txt', 'ss2'],
            ['A', 's3', 'P9855_2090_S113_L004_ac.txt', 'ss3'],
            ['D', 's4', 'P9855_2091_S114_L004_ac.txt', 'ss4'],
            ['E', 's5', 'P9855_2093_S116_L005_ac.txt', 'ss5'],
            ['D', 's6', 'P9855_2096_S119_L005_ac.txt', 'ss6'],
            ['F', 's7', 'P9855_2101_S124_L006_ac.txt', 'ss7'],
            ['F', 's8', 'P9855_2102_S125_L006_ac.txt', 'ss8'],
            ['B', 's9', 'P9855_2104_S127_L006_ac.txt', 'ss9'],
            ['C', 's10', 'P9855_2110_S133_L007_ac.txt', 'ss10'],
            ['C', 's11', 'P9855_2111_S134_L007_ac.txt', 'ss11'],
            ['B', 's12', 'P9855_2112_S135_L007_ac.txt', 'ss12']]

    clones_map_raw = pd.DataFrame(data, columns=['clone', 'tree_id', 'cell_id', 'node_name'])

    clones_map_raw
    return clones_map_raw, data


@app.cell
def __():
    return


@app.cell
def __(clones_map_raw):
    clones_map = dict(clones_map_raw[['tree_id', 'clone']].values)
    all_clones = list(clones_map_raw.clone.unique())
    return all_clones, clones_map


@app.cell
def __(clones_map_raw):
    n_samples_per_clone = clones_map_raw.clone.value_counts()
    n_samples_per_clone
    return n_samples_per_clone,


@app.cell
def __(mo):
    mo.md(r"""### Load info from mt-SCITE runs""")
    return


@app.cell
def __(pd):
    #scite_input_path = f'../../../mt-SCITE/mt-SCITE_output/P9855/stdout/'
    scite_likelihoods = '../../../results/data_analysis/p9855/error_rate_learning/val_scores_normalised.txt'

    #star_tree_likelihoods = '../../../results/data_analysis/p9855/error_rate_learning/val_scores_star_trees.txt'

    repetitions =  [0, 1, 2]
    folds = [0, 1, 2]

    n_trees_file = star_tree_likelihoods = '../../../results/data_analysis/p9855/error_rate_learning/number_of_trees.txt'

    tree_likelihoods = pd.read_csv(scite_likelihoods, index_col=0)
    n_trees = pd.read_csv(n_trees_file, index_col=0)
    #star_tree_likelihoods = pd.read_csv(star_tree_likelihoods)
    return (
        folds,
        n_trees,
        n_trees_file,
        repetitions,
        scite_likelihoods,
        star_tree_likelihoods,
        tree_likelihoods,
    )


@app.cell
def __(pd, tree_likelihoods):
    error_rates = pd.to_numeric(tree_likelihoods.columns)
    error_rates

    len(error_rates)
    return error_rates,


@app.cell
def __():
    return


@app.cell
def __(mo):
    mo.md(r"""### Load info from pmats""")
    return


@app.cell
def __(glob, os):
    pmat_input_path = '../data/P9855_matrix_output/'

    pmats = list(glob(os.path.join(pmat_input_path, '*.csv')))

    pmat_names = []
    n_pmat_rows = []

    for filename in sorted(pmats):
    #for filename in sorted(pmats, key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)]):
        name = os.path.basename(filename).rsplit('.', 1)[0]
        print(name)
        file = open(filename, "r")
        n = len(file.readlines())
        n_pmat_rows.append(n)
        pmat_names.append(name)

    # store in a dict
    pmat_info = {pmat_names[i]: n_pmat_rows[i] for i in range(len(pmat_names))}
    return (
        file,
        filename,
        n,
        n_pmat_rows,
        name,
        pmat_info,
        pmat_input_path,
        pmat_names,
        pmats,
    )


@app.cell
def __(pmats):
    len(pmats)
    return


@app.cell
def __(pmat_info):
    pmat_info
    return


@app.cell
def __(mo):
    mo.md(r"""### Create df containing info from mt-SCITE and pmat""")
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __():
    samples = 13

    #performance = pd.DataFrame(exp_std)
    #performance['lhood'] = lhood
    #performance['lhood'] = performance['lhood'].astype(float)

    #performance['n_trees'] = n_trees
    #performance['error_rate'] = performance[0].str.split('_').str.get(0)

    #performance = performance.rename(columns={0: 'run'})
    #performance = performance.set_index('run')

    #runs = performance.index.tolist()
    #performance = performance.sort_index()

    #performance['n_muts'] = performance['error_rate'].map(pmat_info)

    #performance['entries'] = performance['n_muts']*samples

    #performance['lhood_entries'] = performance['lhood']/performance['entries']
    return samples,


@app.cell
def __(pmat_info):
    pmat_info
    return


@app.cell
def __(tree_likelihoods):
    tree_likelihoods
    return


@app.cell
def __(n_trees, tree_likelihoods):
    # Reshape from wide to long format to integrate with summary data frame
    tree_likelihoods_long = tree_likelihoods.melt(ignore_index=False, 
                                                  var_name="error_rate",
                                        value_name="tree_likelihood").reset_index()
    num_rows = len(tree_likelihoods_long)

    tree_likelihoods_long = tree_likelihoods_long.rename(columns={"index": "fold_idx"})

    tree_likelihoods_long['error_rate'] = tree_likelihoods_long['error_rate'].astype(float)

    # add the repetitions of the k-fold cross validation scheme as indices
    repetition_pattern = (tree_likelihoods_long.index // 3) % 3

    tree_likelihoods_long["rep_idx"] = repetition_pattern



    # same for trees
    n_trees_long = n_trees.melt(ignore_index=False, var_name="error_rate", value_name="n_trees").reset_index()
    n_trees_long = n_trees_long.rename(columns={"index":"fold_idx"})
    n_trees_long['error_rate'] = n_trees_long['error_rate'].astype(float)
    n_trees_long["rep_idx"] = repetition_pattern


    return n_trees_long, num_rows, repetition_pattern, tree_likelihoods_long


@app.cell
def __(tree_likelihoods_long):
    tree_likelihoods_long
    return


@app.cell
def __(n_trees_long):
    n_trees_long
    return


@app.cell
def __(
    error_rates,
    folds,
    itertools,
    n_trees_long,
    pd,
    repetitions,
    tree_likelihoods_long,
):
    # integrate with summary data frame

    combinations = list(itertools.product(error_rates, folds, repetitions))

    df = pd.DataFrame(combinations, columns=['error_rate', 'rep_idx', 'fold_idx'])

    # merge tree likelihoods
    df = df.merge(tree_likelihoods_long, on=['error_rate', 'rep_idx', 'fold_idx'])

    # merge n trees
    df = df.merge(n_trees_long, on=['error_rate', 'rep_idx', 'fold_idx'])

    df['error_rate'] = df['error_rate'].astype(float)
    return combinations, df


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __(df):
    df
    return


@app.cell
def __(mo):
    mo.md(r"""### Read trees""")
    return


@app.cell
def __(MAX_TREES_PER_ERROR_RATE_PER_RUN, nx, os):

    # Constants
    #MAX_TREES_PER_ERROR_RATE_PER_RUN = 100

    def process_trees(df, tree_dir):
        """
        Processes tree files for each row in the DataFrame.
        
        Updates df in-place by setting the number of found trees in column index 4.
        
        Args:
            df (pd.DataFrame): The DataFrame containing error rates and parameters.
            tree_dir (str): Directory containing the tree files.

        Returns:
            dict: A dictionary mapping the run indices (rows) to lists of optimal trees found in that run.
        """
        all_trees = {}

        for row in range(len(df)):
            all_trees[row] = []
            final_idx = 0  # Track how many optimal trees exist per run

            error_rate, rep, fold = df.iloc[row, :3]

            for idx in range(0, MAX_TREES_PER_ERROR_RATE_PER_RUN):
                tree_file = os.path.join(tree_dir, f'learned_{error_rate}_{rep}_{fold}_map{idx}.gv')

                if os.path.exists(tree_file):
                    print(f"Tree {idx} exists for row {row}")

                    try:
                        tree = nx.drawing.nx_pydot.read_dot(tree_file)
                        all_trees[row].append(tree)
                        final_idx = idx  # Update the final valid index
                    except Exception as e:
                        print(f"Error reading {tree_file}: {e}")

            # Update the DataFrame with the final number of trees found
            df.at[row, 4] = final_idx

        return all_trees, df

    # Example usage:
    # df is constructed above
    # mtscite_output_dir = "path_to_tree_files"
    # all_trees = process_trees(df, mtscite_output_dir)

    return process_trees,


@app.cell
def __(mtscite_output_dir, os):
    os.listdir(mtscite_output_dir)
    return


@app.cell
def __(df, mtscite_output_dir, nx, os):
    all_trees = {}
    tree_dir = mtscite_output_dir
    MAX_TREES_PER_ERROR_RATE_PER_RUN = 100

    for row in range(len(df)):
        #print(f"row: {row}")
        all_trees[row] = []
        final_idx = 0  # Track how many optimal trees exist per run

        error_rate_row = df.iloc[row, 0]
        rep = int(df.iloc[row, 1])
        fold = int(df.iloc[row, 2])
        

        for idx in range(0, MAX_TREES_PER_ERROR_RATE_PER_RUN):
            #print(f"index: {idx}")
            tree_file = os.path.join(tree_dir, f'learned_{error_rate_row}_{rep}_{fold}_map{idx}.gv')
            #print(tree_file)

            if os.path.exists(tree_file):
                print(f"Tree {idx} exists for row {row}")

                try:
                    tree = nx.drawing.nx_pydot.read_dot(tree_file)
                    all_trees[row].append(tree)
                    final_idx = idx  # Update the final valid index
                except Exception as e:
                    print(f"Error reading {tree_file}: {e}")
            else:
                break

        # Update the DataFrame with the final number of trees found
        df.at[row, 4] = final_idx
    return (
        MAX_TREES_PER_ERROR_RATE_PER_RUN,
        all_trees,
        error_rate_row,
        final_idx,
        fold,
        idx,
        rep,
        row,
        tree,
        tree_dir,
        tree_file,
    )


@app.cell
def __():
    #all_trees, df_with_tree_number = process_trees(df, mtscite_output_dir)
    return


@app.cell
def __():
    #df_with_tree_number
    return


@app.cell
def __():
    #MAX_TREES_PER_ERROR_RATE_Per_RUN = 100
    #all_trees = {}

    #nrows_df =len(df)
    #for row in range(0, nrows_df):
    #    all_trees[row] = []
    #    n_trees_row = df.iloc[row, 4].astype(int)
        #print(n_trees_row)

    #    error_rate_row = df.iloc[row, 0]
    #    rep = df.iloc[row, 1]
    #    fold = df.iloc[row, 2]
    #    tree_lik_row = df.iloc[row, 3]
        

    #    for idx in range(1, MAX_TREES_PER_ERROR_RATE + 1):
            
    #        tree_file = os.path.join(mtscite_output_dir,
                                    #f'learned_{error_rate_row}_{rep}_{fold}_map{idx}.gv')

     #       if os.path.exists(tree_file):
     #           print(f"Tree {idx} exists for row {row}")
                
     #           tree = nx.drawing.nx_pydot.read_dot(tree_file)
     #           all_trees[row].append(tree)
                
     #       idx = idx +1
            
        #for idx in range(min(MAX_TREES_PER_ERROR_RATE, n_trees_row)):
            #print(n)
        #    error_rate_row = df.iloc[row, 0]
        #    rep = df.iloc[row, 1]
        #    fold = df.iloc[row, 2]
        #    tree_lik_row = df.iloc[row, 3]

        #    if not np.isnan(tree_lik_row):
        #        tree_filename = os.path.join(mtscite_output_dir,
                                        #f'learned_{error_rate_row}_{rep}_{fold}_map{idx}.gv') 
        #        tree = nx.drawing.nx_pydot.read_dot(tree_filename)
                #tree.remove_node('\\n')
        #        all_trees[row].append(tree)
                #print(error_rate)

    #for run, n_tree in zip(runs, n_trees):
    #    all_trees[run] = []
    #    for idx in range(min(MAX_TREES_PER_ERROR_RATE, n)):
    #        #print(n)
    #        tree_filename = f'../../../mt-SCITE/mt-SCITE_output/P9855/{run}_map{idx}.gv' 
    #        tree = nx.drawing.nx_pydot.read_dot(tree_filename)
    #        #tree.remove_node('\\n')
    #        all_trees[run].append(tree)
    #        #print(error_rate)
    return


@app.cell
def __():
    return


@app.cell
def __(all_clones, nx, pd):
    def new_count_row(all_clone_names, clone_name=None):
        """ Create a DataFrame row with one entry per clone name. 

        If `clone_name` is not None, initialize that entry to 1.
        """
        row = pd.Series(data=0, index=all_clone_names)
        if clone_name is not None:
            row[clone_name] = 1
        return row

    def dfs_clones_count(g, clones, all_clone_names, source_node):
        clones_count = pd.DataFrame({sample_name: new_count_row(all_clones, clone_name) for sample_name, clone_name in clones.items()})
        for pre_node, post_node, edge_label in nx.dfs_labeled_edges(g, source_node):
            if edge_label is not 'reverse' or pre_node == post_node: 
                continue
            pre_node_counter = clones_count.get(pre_node, new_count_row(all_clones))
            if post_node not in clones_count.columns:
                print('!! possible mutation in a leaf: not found', post_node)
                continue
            clones_count[pre_node] = pre_node_counter + clones_count[post_node]

        return clones_count

    def purity(clones_count):
        return clones_count.div(clones_count.sum(axis=0), axis=1)

    def get_root_node(t):
        root_nodes = [n for n,d in t.in_degree() if d==0]
        assert len(root_nodes) == 1
        root_node = root_nodes[0]
        return root_node
    return dfs_clones_count, get_root_node, new_count_row, purity


@app.cell
def __(mo):
    mo.md(r"""# Avg purity of nodes with >1 sample""")
    return


@app.cell
def __(all_clones):
    all_clones
    return


@app.cell
def __():
    #G = all_trees[18][0]
    return


@app.cell
def __(G, nx, plt):
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)  # Positioning nodes for visualization
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=500, font_size=10)
    plt.title("MultiDiGraph with 28 Nodes and 27 Edges")
    plt.show()
    return pos,


@app.cell
def __(
    all_clones,
    all_trees,
    clones_map,
    df,
    dfs_clones_count,
    get_root_node,
    np,
    nx,
    purity,
):
    import warnings
    warnings.filterwarnings('ignore')

    df['avg_purity_more_than_one_sample'] = 0
    df['frac_purity_100_more_than_one_sample'] = 0
    for row_df, trees in all_trees.items():

        error_rate = df.iloc[row_df, 0]
        print(f'performing purity calculation for error rate {error_rate}')
        avg_purities = []
        frac_purities_100 = []
        #n_purities_100 = []
        for t in trees:

            if not nx.MultiDiGraph.is_multigraph(t):
                print(t)
                df.loc[row_df, 'avg_purity_more_than_one_sample'] = np.nan
                df.loc[row_df, 'frac_purity_100_more_than_one_sample'] = np.nan
            else:

                print(error_rate, get_root_node(t))
                clones_count = dfs_clones_count(t, clones_map, all_clones, source_node=get_root_node(t))
                print("Printing clones count")
                print(clones_count)
                more_than_one_sample = clones_count.sum(axis=0) > 1
                print("More than one sample")
                print(more_than_one_sample)
                p = purity(clones_count).loc[:, more_than_one_sample]
                print(f"printing p: {p}")

                # exclude mutations shared by all clones
                p = p.T
                p = p.mask((p.A > 0) & (p.B > 0) & (p.C > 0) & (p.D > 0) & (p.E > 0) & (p.F > 0))
                p = p.dropna().T


                # Average purity
                avg_purity = p.max(axis=0).mean()
                avg_purities.append(avg_purity)
                print(avg_purities)
                # Fraction of nodes with 100% purity
                is_purity_100 = np.isclose(p.max(axis=0), 1.0)
                frac_purity_100 = is_purity_100.sum() / is_purity_100.shape[0]
                frac_purities_100.append(frac_purity_100)

                
                df.loc[row_df, 'avg_purity_more_than_one_sample'] = avg_purities[0]
                df.loc[row_df, 'frac_purity_100_more_than_one_sample'] = frac_purities_100[0]
        #performance.loc[error_rate, 'avg_purity_more_than_one_sample'] = avg_purities[0]
        #performance.loc[error_rate, 'frac_purity_100_more_than_one_sample'] = frac_purities_100[0]
    return (
        avg_purities,
        avg_purity,
        clones_count,
        error_rate,
        frac_purities_100,
        frac_purity_100,
        is_purity_100,
        more_than_one_sample,
        p,
        row_df,
        t,
        trees,
        warnings,
    )


@app.cell
def __():
    #cond = performance['error_rate'] == '0.0007'
    #performance[cond]
    return


@app.cell
def __(df, pd):
    #pd.set_option('display.max_rows', None)
    pd.options.display.max_rows = 1000
    df
    #performance
    return


@app.cell
def __(df):
    len(df)
    return


@app.cell
def __():
    return


@app.cell
def __(df):
    df['avg_purity_more_than_one_sample']
    return


@app.cell
def __(df):
    df.iloc[20:30, ]
    return


@app.cell
def __(df):
    max_index = df['avg_purity_more_than_one_sample'].idxmax()
    max_index
    return max_index,


@app.cell
def __(df, max_index):
    df.loc[max_index]
    return


@app.cell
def __(mo):
    mo.md(r"""# Plots""")
    return


@app.cell
def __():
    #source.from_file(f'../../../mt-SCITE/mt-SCITE_output/P9855/0.0008_1_map0.gv')
    return


@app.cell
def __():
    # Plot a tree
    # Source.from_file(f'../../../mt-SCITE/mt-SCITE_output/P9855_mrate_1000/0.0005_10_map0.gv')
    return


@app.cell
def __():
    #pd.set_option('display.max_rows', None)
    #performance
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __(mo):
    mo.md(r"""## Tree Likelihood""")
    return


@app.cell
def __(df):
    # Compute the mean tree likelihood per error rate
    mean_likelihood_df = df.groupby("error_rate", as_index=False)["tree_likelihood"].mean()
    mean_likelihood_df = mean_likelihood_df.dropna(subset=['tree_likelihood'])
    return mean_likelihood_df,


@app.cell
def __(mean_likelihood_df):
    mean_likelihood_df
    return


@app.cell
def __(mean_likelihood_df):
    best_likelihood = mean_likelihood_df['tree_likelihood'].min()
    print(f'The best average likelihood across all folds and repetitions is {best_likelihood}')
    best_likelihood_idx = mean_likelihood_df['tree_likelihood'].idxmin()
    best_error_rate = mean_likelihood_df.iloc[best_likelihood_idx, 0]
    print(f'The corresponding error rate is {best_error_rate}')
    best_error_rate
    return best_error_rate, best_likelihood, best_likelihood_idx


@app.cell
def __(best_error_rate, df):
    # The box plot uses categorical values of the x axis while I want to use the numerical value of the error rate to plot an additional horizontal line. 
    # Find the index of best_error_rate in the sorted unique error rates

    error_rate_order = sorted(df["error_rate"].unique())  # Get sorted unique error rates
    best_error_rate_idx = error_rate_order.index(best_error_rate)  # Get position index

    return best_error_rate_idx, error_rate_order


@app.cell
def __(error_rate_order):
    error_rate_order[23:]
    return


@app.cell
def __(error_rate_order):
    # Define the subset of error rates to display
    selected_error_rates = [0.0001, 0.0008, 0.0015, 0.0031, 0.0101, 0.0181, 0.0231]
    tick_positions = [error_rate_order.index(er) for er in selected_error_rates if er in error_rate_order]

    return selected_error_rates, tick_positions


@app.cell
def __(
    best_error_rate_idx,
    df,
    plot_style,
    plt,
    selected_error_rates,
    sns,
    tick_positions,
):
    df_plot = df.dropna(subset=['tree_likelihood'])
    with plot_style(figsize=(5, 3), ticklabelsize=8, labelsize=10):
        sns.boxplot(data=df_plot, x="error_rate", y="tree_likelihood", color="lightgray", showfliers=False) #marker='o'
        
        plt.axvline(x=best_error_rate_idx, color='red', linestyle='dashed', linewidth=1.5, label="Best Likelihood")

        # Customize X-axis ticks to show only selected values
        plt.xticks(ticks=tick_positions, labels=[str(er) for er in 
                                                 selected_error_rates], 
                   rotation=90)
        #plt.xticks(rotation=90)
        plt.ylabel('Normalized tree likelihood')
        plt.xlabel('Error rate')

        #plt.show()
        plt.savefig(f'../figures/fig3/likelihood.svg', dpi=300, bbox_inches='tight', transparent=True)
    return df_plot,


@app.cell
def __():
    #with plot_style(figsize=(40, 5), ticklabelsize=10, labelsize=10):
    #    sns.lineplot(data=performance, x="error_rate", y="lhood_entries", lw=1) #marker='o'
    #    plt.grid(axis='both')
        #ax.set_xlim(0, 20)
    #    plt.xticks(rotation=90)
        #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #    plt.ylabel('Best likelihood / matrix entries')
    #    plt.xlabel('Error rate')

    #plt.savefig(f'../../data/P9855_figures/lhood_entries.svg', dpi=300, bbox_inches='tight', transparent=True)
    #plt.savefig(f'../../data/P9855_figures/lhood_entries.jpg', dpi=300, bbox_inches='tight', transparent=True)
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __(mo):
    mo.md(r"""## Average purity""")
    return


@app.cell
def __():
    return


@app.cell
def __(df, plot_style, plt, sns):
    with plot_style(figsize=(4.7, 2.5), ticklabelsize=10, labelsize=10): #4.7
        sns.lineplot(data=df, x="error_rate", y="avg_purity_more_than_one_sample", lw=1, color='black')
        plt.grid(axis='both')
        plt.xticks(rotation=90)
        plt.ylim(0, 1)
        plt.xticks(fontsize=5)

        #plt.xlim(0.0001, 0.0008)
        plt.ylabel('Purity')
        plt.xlabel('Error rate')
        plt.show()


        #plt.savefig(f'../../data/P9855_figures/purity.pdf', dpi=300, bbox_inches='tight', transparent=True)
    return


@app.cell
def __():
    return


@app.cell
def __():
    #error_rate_values = performance['error_rate']

    # Calculate the number of labels you want (in this case, 10)
    #num_labels = 100

    # Calculate the step size to evenly distribute the labels
    #step_size = len(error_rate_values) // num_labels

    # Get the indices for the x-axis ticks
    #x_tick_indices = np.arange(0, len(error_rate_values), step_size)

    # Get the corresponding error_rate values for the x-axis labels
    #x_tick_labels = error_rate_values.iloc[x_tick_indices]

    # Your plot code here
    #with plot_style(figsize=(20, 2.7), ticklabelsize=10, labelsize=10):
    ##    sns.lineplot(data=performance, x="error_rate", y="avg_purity_more_than_one_sample", lw=1, color='black')
    #    plt.grid(axis='both')
    #    plt.xticks(x_tick_indices, x_tick_labels, rotation=90)  # Set the custom tick positions and labels
    #    plt.ylabel('Purity')
    #    plt.xlabel('Error rate')

    # Show or save the plot
    #plt.show()
    return


@app.cell
def __():
    return


@app.cell
def __(mo):
    mo.md(r"""## Number of mutations""")
    return


@app.cell
def __(pd, plot_style, plt, pmat_info, sns):
    performance = pd.DataFrame(list(pmat_info.items()), columns=["error_rate", "n_muts"])


    with plot_style(figsize=(10, 2.5), ticklabelsize=10, labelsize=10):
        sns.lineplot(data=performance, x="error_rate", y="n_muts", lw=1, color='Black') #marker='o'
        plt.grid(axis='both')
        #ax.set_xlim(0, 20)
        #plt.yscale("log")
        plt.xticks(rotation=90)
        #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.ylabel('Number of mutations')
        plt.xlabel('Error rate')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # Get the x-axis values and labels
        #x_values = performance["error_rate"]
        #x_labels = performance["error_rate"]

        # Plot only every 10th x-axis label
        #step = 200
        #plt.xticks(x_values[::step], x_labels[::step])

        plt.show()
        #plt.savefig(f'../../data/P9855_figures/n_muts.svg', dpi=300, bbox_inches='tight', transparent=True)
        #plt.savefig(f'../../data/P9855_figures/n_muts.jpg', dpi=300, bbox_inches='tight', transparent=True)
    return performance,


@app.cell
def __(mo):
    mo.md(r"""## Number of trees""")
    return


@app.cell
def __(performance):
    performance.rename(columns={'n trees': 'n_trees'}, inplace=True)
    return


@app.cell
def __(performance):
    performance
    return


@app.cell
def __(df, plot_style, plt, sns):
    with plot_style(figsize=(10, 10), ticklabelsize=10, labelsize=10):

        sns.lineplot(data=df.reset_index(), x='error_rate', y="n_trees")

        plt.grid(axis='both')
        #plt.yscale('log')
        plt.xlabel('Error rate')
        plt.ylabel('Number of trees')
        plt.xticks(rotation = 90)


        # Get the x-axis values and labels
        #x_values = performance["error_rate"]
        #x_labels = performance["error_rate"]

        # Plot only every 250th x-axis label
        #step = 200
        #plt.xticks(x_values[::step], x_labels[::step])

        plt.show()

        #plt.savefig(f'../../data/P9855_figures/n_trees.svg', dpi=300, bbox_inches='tight', transparent=True)
        #plt.savefig(f'../../data/P9855_figures/n_trees.jpg', dpi=300, bbox_inches='tight', transparent=True)
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __():
    import marimo as mo
    return mo,


if __name__ == "__main__":
    app.run()
