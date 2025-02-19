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
           'axes.spines.top': True,
           'axes.spines.right': True,
           'axes.spines.left': True,
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
    mtscite_output_dir = '../../../results/data_analysis/yfv2001/error_learning_k3_r3_mg7/'
    return mtscite_output_dir,


@app.cell
def __(mo):
    mo.md(r"""### Load likelihoods generated during error learning""")
    return


@app.cell
def __(mtscite_output_dir, os, pd):
    #scite_input_path = f'../../../mt-SCITE/mt-SCITE_output/P9855/stdout/'
    mtscite_likelihoods = os.path.join(mtscite_output_dir, 'val_scores.txt')

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

    #df = df.dropna(subset=['tree_likelihood'])
    return combinations, df


@app.cell
def __(mo):
    mo.md(r"""## Plot""")
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
    #selected_error_rates = [0.0001, 0.0008, 0.0015, 0.0031, 0.0101, 0.0171, 0.0241, 0.0321, 0.0351]
    tick_positions = [error_rate_order.index(er) for er in error_rate_order]
    return tick_positions,


@app.cell
def __(
    best_error_rate_idx,
    df,
    error_rate_order,
    plot_style,
    plt,
    sns,
    tick_positions,
):
    # Show the error rates that are not NaN
    with plot_style(figsize=(4.4, 2.5), ticklabelsize=11, labelsize=12):
        sns.boxplot(data=df, x="error_rate", y="tree_likelihood", color="lightgray", showfliers=False) #marker='o'

        # add estimated error rate
        plt.axvline(x=best_error_rate_idx, color='red', linestyle='dashed', linewidth=1.5, label="Best Likelihood")

        # Customize X-axis ticks to show only selected values
        plt.xticks(ticks=tick_positions, labels=[str(er) for er in 
                                                 error_rate_order], 
                   rotation=90)
        plt.ylabel('Normalized tree likelihood')
        plt.xlabel('Error rate')
        #ax = plt.gca()
        #ax.xaxis.set_tick_params(pad=-6)

        plt.show()
        #plt.savefig(f'../figures/fig3/likelihood.svg', dpi=300, bbox_inches='tight', transparent=True)
    return


@app.cell
def __(df, np):
    # Rationale here is to plot the likelihoods and report the NAN values in the plot
    valid_df = df[df['tree_likelihood'].notnull()]

    invalid_df = df.copy()
    invalid_df['tree_likelihood'] = np.where(
        invalid_df['tree_likelihood'].isnull(),  # True if originally NaN
        -0.05,                                     # constant y-value for invalid points
        np.nan                                   # otherwise set to NaN so they're not plotted here
    )
    return invalid_df, valid_df


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __(
    best_error_rate_idx,
    df,
    error_rate_order,
    invalid_df,
    plot_style,
    plt,
    tick_positions,
):
    # Plot likelihoods and error rates
    with plot_style(figsize=(4.4, 2.5), ticklabelsize=12, labelsize=12):


        # 1) Create figure & axes
        fig, ax = plt.subplots(figsize=(4.4, 2.5))

        # 2) Prepare data for boxplot
        box_data = []
        for c in error_rate_order:
            subset = df.loc[df["error_rate"] == c, "tree_likelihood"]
            box_data.append(subset.dropna().values)  # dropna -> valid data only

        # 3) Plot the boxplot with patch_artist=True so we can color the boxes
        bp = ax.boxplot(
            box_data,
            positions=range(len(error_rate_order)),
            patch_artist=True,  # needed for facecolor/edgecolor changes
            showfliers=False,
            widths=0.6
        )

        # 4) Style the boxplot lines/faces
        for element in ['whiskers', 'caps', 'medians']:
            plt.setp(bp[element], color='black')  # or 'gray'

        for box in bp['boxes']:
            box.set_facecolor('lightgray')   # fill color for the boxes
            box.set_edgecolor('black')       # outline color

        # 5) Overlay invalid points at the same integer positions
        for i, e in enumerate(error_rate_order):
            sub = invalid_df[invalid_df["error_rate"] == e]
            yvals = sub["tree_likelihood"]
            ax.scatter(
                [i] * len(yvals),  # all plotted at x = i
                yvals,
                marker='o',
                s=10,
                linewidths=0.5,
                facecolors='none',  # open circle
                edgecolors='black',
                label='NaN (No valid tree)' if i == 0 else ''  # label once
            )

        # 6) Customize x-axis
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(er) for er in error_rate_order], rotation=90)
        ax.set_xlabel("Error rate")
        ax.set_ylabel("Normalized tree likelihood")

        # 7) Add vertical line for best likelihood
        ax.axvline(
            x=best_error_rate_idx,  # must be an integer index into error_rates_order
            color='red',
            linestyle='dashed',
            linewidth=1.5,
            label="Best Likelihood"
        )
        #ax = plt.gca()
        ax.xaxis.set_tick_params(pad=-4)

        #ax.legend()
        #plt.tight_layout()
        #plt.show()
        plt.savefig(f'../figures/fig4/likelihood_nan.svg', dpi=300, bbox_inches='tight', transparent=True)
    return ax, box, box_data, bp, c, e, element, fig, i, sub, subset, yvals


@app.cell
def __():
    import marimo as mo
    return mo,


if __name__ == "__main__":
    app.run()
