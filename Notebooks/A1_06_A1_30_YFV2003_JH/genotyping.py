import numpy as np
import pandas as pd
from scipy.stats import multinomial


NUCLEOTIDES = ['A', 'C', 'G', 'T']
COUNTS_COLUMNS = ['Count_A', 'Count_C', 'Count_G', 'Count_T']
#FREQ_COLUMNS = ['Freq_A', 'Freq_C', 'Freq_G', 'Freq_T']
PROB_COLUMNS = ['Prob_mutation_A', 'Prob_mutation_C',
                'Prob_mutation_G', 'Prob_mutation_T']


#def allele_frequency(cell_counts):
#    """ Compute allele frequency for a cell count matrix.
#
#    Parameters
#    ----------
#    cell_counts : pandas.DataFrame
#        DataFrame with columns `'CHR', 'POS', 'Count_A', 'Count_C',
#        'Count_G', 'Count_T', 'Good_depth'`, and one row per position.
#
#    Returns
#    -------
#    af : pandas.DataFrame
#        DataFrame with allele frequencies for each position.
#
#    """
#    counts_per_position = cell_counts[COUNTS_COLUMNS]
#    tot_counts_per_position = counts_per_position.sum(axis=1)
#    frequencies = counts_per_position.div(tot_counts_per_position, axis=0)
#    frequencies.columns = FREQ_COLUMNS
#    af = pd.concat([cell_counts[['#CHR', 'POS']], frequencies], axis=1)
#    return af


def _log_prob_ABC_no_mutation(
        reads_R, reads_A, reads_B, reads_C, error_rate):
    """ P(reads_A, reads_B, reads_C | reads_R, no mutation)
    It does not matter which nucleotides are A, B, and C
    Parameters
    ----------
    reads_R : int
        Number of reads for the reference nucleotide
    reads_A, reads_B, reads_C : int
        Number of reads for the non-reference nucleotides
    Returns
    -------
    log_prob_ABC : float
        log P(reads_A, reads_B, reads_C | reads_R, no mutation)
    """
    total_reads = reads_R + reads_A + reads_B + reads_C
    log_prob_ABC = multinomial.logpmf(
       np.dstack([reads_A.tolist(), reads_B.tolist(),
                  reads_C.tolist(), reads_R.tolist()]).squeeze(),
       n=total_reads,
       p=[error_rate / 3, error_rate / 3, error_rate / 3, 1 - error_rate] # p interpretation of rho
    )
    return log_prob_ABC


def _log_prob_ABC_with_mutation_at_A(
        reads_R, reads_A, reads_B, reads_C, error_rate):
    """ Compute P(reads_A, reads_B, reads_C | reads_R, mutation at A).
    A is always the mutated nucleotide.
    R is the reference nucleotide.
    B and C are the remaining nucleotides (which ones it does not matter)
    !!! This distribution assumes that reads_A is always > 0 !!!
    Parameters
    ----------
    reads_R : int
        Number of reads for the reference nucleotide
    reads_A, reads_B, reads_C : int
        Number of reads for the non-reference nucleotides
    Returns
    -------
    log_prob_ABC : float
        log P(reads_A, reads_B, reads_C | reads_R, no mutation)
    """

    total_reads = reads_R + reads_A + reads_B + reads_C
    reads_RBC = reads_R + reads_B + reads_C

    zero_sum_mask = reads_RBC == 0.0

    log_prob_BC = multinomial.logpmf(
        np.dstack([reads_B.tolist(),
                   reads_C.tolist(),
                   reads_R.tolist()]).squeeze(),
        n=reads_RBC,
        p=[error_rate / 2.0, error_rate / 2.0, 1 - error_rate] # p interpretation of rho
    )
    log_prob_BC[zero_sum_mask] = 0.0

    log_prob_ABC = log_prob_BC - np.log(total_reads)

    return log_prob_ABC


def nucleotide_mutation_prob(
        cell_counts, reference,
        error_rate_when_no_mutation,
        error_rate_when_mutation,
        p_mutation):
    """ Compute P(mutation at X | read counts).
    Parameters
    ----------
    cell_counts : pd.DataFrame
        DataFrame with columns `'CHR', 'POS', 'Count_A', 'Count_C',
        'Count_G', 'Count_T', 'Good_depth'`, and one row per position.
        (`'CHR'`, `'POS'`, and `'Good_depth'` are ignored)
    reference : pd.Series
        Series containing the name of the reference nucleotide (`'A'`,
        `'C'`, `'G'`, `'T'`, or `'N'` for unknown)
    error_rate_when_no_mutation : float
        Experimental error rate when there is no mutation
    error_rate_when_mutation : float
        Experimental error rate when there is a mutation at one of the
        non-reference nucleotide
    p_mutation : float
        Probability of a mutation at one position
    Returns
    -------
    prob_mutation : pd.DataFrame
        DataFrame with probability of mutation at each nucleotide
        for each position.
    """

    prob_mutation = cell_counts[['#CHR', 'POS']].copy()
    for col in PROB_COLUMNS:
        #prob_mutation[col] = pd.np.nan
        prob_mutation[col] = np.nan 

    # Positions with missing reference
    missing_ref_mask = reference == 'N'
    prob_mutation.loc[missing_ref_mask, PROB_COLUMNS] = np.nan

    # Positions with zero reads
    zero_reads_mask = cell_counts[COUNTS_COLUMNS].sum(axis=1) == 0
    prob_mutation.loc[zero_reads_mask, PROB_COLUMNS] = np.nan

    # Explicit list of all reference/non-reference columns combinations
    REFS_MAP = {
        'A': ('Count_A', ['Count_C', 'Count_G', 'Count_T']),
        'C': ('Count_C', ['Count_A', 'Count_G', 'Count_T']),
        'G': ('Count_G', ['Count_A', 'Count_C', 'Count_T']),
        'T': ('Count_T', ['Count_A', 'Count_C', 'Count_G']),
    }

    # Compute in blocks of rows having each successive nucleotide as reference
    for ref in NUCLEOTIDES:
        reference_mask = (reference == ref) & (~zero_reads_mask)
        # No reference counts for this nucleotide
        if reference_mask.sum() == 0:
            continue

        # Extract reference and non-reference columns
        ref_count_col, non_ref_count_cols = REFS_MAP[ref]
        ref_count = cell_counts.loc[reference_mask, ref_count_col]
        non_ref_count = cell_counts.loc[reference_mask, non_ref_count_cols]

        # No mutation, M_ij = R
        # eq. 10
        cell_counts_at_pos_ABC = non_ref_count
        log_p_v_ABC_given_A = _log_prob_ABC_no_mutation(
            reads_R=ref_count,
            reads_A=cell_counts_at_pos_ABC.iloc[:, 0],
            reads_B=cell_counts_at_pos_ABC.iloc[:, 1],
            reads_C=cell_counts_at_pos_ABC.iloc[:, 2],
            error_rate=error_rate_when_no_mutation,
        )
        p_v_ABC_given_A = np.exp(log_p_v_ABC_given_A)
        # eq. 10.5
        p_M_eq_A = 1.0 - p_mutation

        unnorm_P_M_eq_A_given_v_ABC = p_v_ABC_given_A * p_M_eq_A
        prob_col = f'Prob_mutation_{ref}'
        prob_mutation.loc[reference_mask, prob_col] = (
            unnorm_P_M_eq_A_given_v_ABC
        )

        # Mutation, M_ij = A
        # eq. 8 and 9
        for col_A in non_ref_count_cols:
            cell_counts_at_pos_BC = non_ref_count.drop(col_A, axis=1)
            log_p_v_ABC_given_A = _log_prob_ABC_with_mutation_at_A(
                reads_R=ref_count,
                reads_A=non_ref_count[col_A],
                reads_B=cell_counts_at_pos_BC.iloc[:, 0],
                reads_C=cell_counts_at_pos_BC.iloc[:, 1],
                error_rate=error_rate_when_mutation,
            )
            p_v_ABC_given_A = np.exp(log_p_v_ABC_given_A)
            # eq. 11
            p_M_eq_A = 1.0 / 3.0 * p_mutation
            unnorm_P_M_eq_A_given_v_ABC = p_v_ABC_given_A * p_M_eq_A
            prob_col = f'Prob_mutation_{col_A[-1]}'
            prob_mutation.loc[reference_mask, prob_col] = (
                unnorm_P_M_eq_A_given_v_ABC
            )

    # Normalize the probability distribution of each row
    prob_mutation.loc[:, PROB_COLUMNS] = (
        prob_mutation.loc[:, PROB_COLUMNS].div(
            prob_mutation.loc[:, PROB_COLUMNS].sum(axis=1),
            axis=0
        )
    )
    return prob_mutation


def mutation_prob(nucleotide_mutation_prob, reference):
    """ Compute P(mutation | read counts) given P(mutation at X | read counts).
    Parameters
    ----------
    nucleotide_mutation_prob : pd.DataFrame
        DataFrame with columns `'CHR', 'POS', 'Prob_mutation_A',
        'Prob_mutation_C', 'Prob_mutation_G', 'Prob_mutation_T',
        and one row per position.
        This is typically the output of `nucleotide_mutation_prob`.
        (`'CHR'` and `'POS'` are ignored)
    reference : pd.Series
        Series containing the name of the reference nucleotide (`'A'`,
        `'C'`, `'G'`, `'T'`, or `'N'` for unknown)
    Returns
    -------
    prob_mutation : pd.DataFrame
        DataFrame with probability of mutation at each position.
    """

    # Rename columns for fast indexing through reference
    p = (
        nucleotide_mutation_prob
        .rename(columns={f'Prob_mutation_{n}': n for n in NUCLEOTIDES})
        .loc[:, NUCLEOTIDES]
    )
    # Handle unknown references
    p['N'] = np.nan

    prob_mutation = nucleotide_mutation_prob[['#CHR', 'POS']].copy()
    prob_mutation['Prob_mutation'] = 1 - p.lookup(p.index, reference.values)
    

    return prob_mutation