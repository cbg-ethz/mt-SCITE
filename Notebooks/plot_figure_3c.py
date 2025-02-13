import marimo

__generated_with = "0.8.3"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(
        """
        **Author Note**  
        Date: 20215-02-13
        Author: Sophie Seidel
        Purpose: 
        1) Find the best error rate for ATAC-seq data given the 3-fold cross validation analysis and
        2) Generate a plot showing the normalised tree likelihoods across error rates
        """
    )
    return


@app.cell
def __():
    # import libs
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


    return (
        Counter,
        EngFormatter,
        FormatStrFormatter,
        Markdown,
        Source,
        display,
        glob,
        itertools,
        np,
        numbers,
        nx,
        os,
        pd,
        plt,
        re,
        sns,
        stats,
    )


@app.cell
def __(plt, sns):
    # plot settings

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
    return blue, plot_style


@app.cell
def __():
    # specify directories
    mtscite_output_dir = '../../../results/data_analysis/p9855/error_rate_learning/'
    return mtscite_output_dir,


@app.cell
def __(mo):
    mo.md(r"""### Load likelihoods generated during error learning""")
    return


@app.cell
def __(mtscite_output_dir, os, pd):
    #scite_input_path = f'../../../mt-SCITE/mt-SCITE_output/P9855/stdout/'
    mtscite_likelihoods = os.path.join(mtscite_output_dir, 'val_scores_normalised.txt')

    # we used 3 repetitions for every 3-fold cross validation
    repetitions =  [0, 1, 2]
    folds = [0, 1, 2]

    tree_likelihoods = pd.read_csv(mtscite_likelihoods, index_col=0)

    return folds, mtscite_likelihoods, repetitions, tree_likelihoods


@app.cell
def __(pd, tree_likelihoods):
    error_rates = pd.to_numeric(tree_likelihoods.columns)
    error_rates

    len(error_rates)
    return error_rates,


@app.cell
def __(mo):
    mo.md("""### Reshape data before plotting""")
    return


@app.cell
def __(tree_likelihoods):
    # Reshape from wide to long format to integrate with fold and repetition info
    tree_likelihoods_long = tree_likelihoods.melt(ignore_index=False, 
                                                  var_name="error_rate",
                                        value_name="tree_likelihood").reset_index()
    num_rows = len(tree_likelihoods_long)

    tree_likelihoods_long = tree_likelihoods_long.rename(columns={"index": "fold_idx"})

    tree_likelihoods_long['error_rate'] = tree_likelihoods_long['error_rate'].astype(float)

    # add the repetitions of the k-fold cross validation scheme as indices
    repetition_pattern = (tree_likelihoods_long.index // 3) % 3

    tree_likelihoods_long["rep_idx"] = repetition_pattern



    return num_rows, repetition_pattern, tree_likelihoods_long


@app.cell
def __(
    error_rates,
    folds,
    itertools,
    pd,
    repetitions,
    tree_likelihoods_long,
):
    # integrate with summary data frame
    # generate combinations of error rates; 3-fold and 3 repetition indices
    combinations = list(itertools.product(error_rates, folds, repetitions))

    df = pd.DataFrame(combinations, columns=['error_rate', 'rep_idx', 'fold_idx'])

    # merge tree likelihoods
    df = df.merge(tree_likelihoods_long, on=['error_rate', 'rep_idx', 'fold_idx'])
    df['error_rate'] = df['error_rate'].astype(float)

    # drop likelihoods and error rates, where there the likelihood was NA. This is the output from mt-scite, if the number of mutations is <10, where an error rate is not learned

    df = df.dropna(subset=['tree_likelihood'])
    return combinations, df


@app.cell
def __(mo):
    mo.md(r"""## Plot """)
    return


@app.cell
def __(df):
    # Compute the mean tree likelihood per error rate across repetitions and folds
    mean_likelihood_df = df.groupby("error_rate", as_index=False)["tree_likelihood"].mean()

    return mean_likelihood_df,


@app.cell
def __(mean_likelihood_df):
    mean_likelihood_df
    return


@app.cell
def __(mean_likelihood_df):
    best_likelihood = mean_likelihood_df['tree_likelihood'].min()
    print(f'The best average likelihood is {best_likelihood}')
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
def __():
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

    with plot_style(figsize=(5, 3), ticklabelsize=8, labelsize=10):
        sns.boxplot(data=df, x="error_rate", y="tree_likelihood", color="lightgray", showfliers=False) #marker='o'

        # add estimated error rate
        plt.axvline(x=best_error_rate_idx, color='red', linestyle='dashed', linewidth=1.5, label="Best Likelihood")

        # Customize X-axis ticks to show only selected values
        plt.xticks(ticks=tick_positions, labels=[str(er) for er in 
                                                 selected_error_rates], 
                   rotation=90)
        plt.ylabel('Normalized tree likelihood')
        plt.xlabel('Error rate')

        #plt.show()
        plt.savefig(f'../figures/fig3/likelihood.svg', dpi=300, bbox_inches='tight', transparent=True)
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
