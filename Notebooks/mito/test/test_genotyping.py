import numpy as np
from numpy.testing import assert_almost_equal
import pandas as pd
from pandas.testing import assert_frame_equal
from scipy import stats

from mito.genotyping import (
    _log_prob_ABC_no_mutation,
    _log_prob_ABC_with_mutation_at_A,
    allele_frequency,
    nucleotide_mutation_prob,
    PROB_COLUMNS,
    mutation_prob,
)


def test_allele_frequency():
    cell_counts = pd.DataFrame(
        data={
            '#CHR': ['MT', 'MT'],
            'POS': [1, 2],
            'Count_A': [1000, 800],
            'Count_C': [0, 400],
            'Count_G': [0, 600],
            'Count_T': [0, 200],
            'Good_depth': [1000, 2000],
        },
    )

    af = allele_frequency(cell_counts)

    expected = pd.DataFrame(
        data={
            '#CHR': ['MT', 'MT'],
            'POS': [1, 2],
            'Freq_A': [1.0, 0.4],
            'Freq_C': [0.0, 0.2],
            'Freq_G': [0.0, 0.3],
            'Freq_T': [0.0, 0.1],
        },
    )
    assert_frame_equal(expected, af, check_like=True)


def test__log_prob_ABC_no_mutation():
    error_rate = 0.096
    log_prob_reads_ABC_given_R = _log_prob_ABC_no_mutation(
        reads_R=np.array([100]),
        reads_A=np.array([12]),
        reads_B=np.array([34]),
        reads_C=np.array([7]),
        error_rate=error_rate,
    )
    expected_prob = stats.multinomial.pmf(
        [100, 12, 34, 7], 100 + 12 + 34 + 7,
        [1 - error_rate, error_rate / 3, error_rate / 3, error_rate / 3]
    )
    assert_almost_equal(log_prob_reads_ABC_given_R, np.log(expected_prob))


def test__log_prob_ABC_no_mutation_is_normalized():
    error_rate = 0.005
    total_reads = 13

    tot_prob = 0.0
    # !!! Note that we start at `a > 0`: the case with a mutation in A is
    # only well-defined if there is at least one read at A
    for a in range(total_reads + 1):
        for b in range(total_reads - a + 1):
            for c in range(total_reads - a - b + 1):
                r = total_reads - a - b - c
                log_prob = _log_prob_ABC_no_mutation(
                    reads_R=np.array([r]),
                    reads_A=np.array([a]),
                    reads_B=np.array([b]),
                    reads_C=np.array([c]),
                    error_rate=error_rate,
                )
                tot_prob += np.exp(log_prob)
    assert_almost_equal(tot_prob, 1.0)


def test__log_prob_ABC_with_mutation_at_A():
    error_rate = 0.021
    log_prob_reads_ABC_given_B = _log_prob_ABC_with_mutation_at_A(
        reads_R=np.array([12]),
        reads_A=np.array([100]),
        reads_B=np.array([34]),
        reads_C=np.array([7]),
        error_rate=error_rate,
    )
    expected_prob = stats.multinomial.pmf(
        [12, 34, 7], 12 + 34 + 7,
        [1 - error_rate, error_rate / 2, error_rate / 2]
    ) * 1.0 / (100 + 12 + 34 + 7)
    assert_almost_equal(log_prob_reads_ABC_given_B, np.log(expected_prob))


def test__log_prob_ABC_with_mutation_at_A_is_normalized():
    error_rate = 0.005
    total_reads = 17

    tot_prob = 0.0
    # !!! Note that we start at `a > 0`: the case with a mutation in A is
    # only well-defined if there is at least one read at A
    for a in range(1, total_reads + 1):
        for b in range(total_reads - a + 1):
            for c in range(total_reads - a - b + 1):
                r = total_reads - a - b - c
                log_prob = _log_prob_ABC_with_mutation_at_A(
                    reads_R=np.array([r]),
                    reads_A=np.array([a]),
                    reads_B=np.array([b]),
                    reads_C=np.array([c]),
                    error_rate=error_rate,
                )
                tot_prob += np.exp(log_prob)
    assert_almost_equal(tot_prob, 1.0)


def test_nucleotide_mutation_prob_deterministic():
    # Test the `nucleotide_mutation_prob` function in the simple case where
    # all the reads correspond to a single nucleotide.

    N = 1000
    cell_counts = pd.DataFrame(
        data={
            '#CHR': ['MT', 'MT', 'MT', 'MT'],
            'POS': [1, 2, 3, 4],
            'Count_A': [N, 0, 0, 0],
            'Count_C': [0, N, 0, 0],
            'Count_G': [0, 0, N, 0],
            'Count_T': [0, 0, 0, N],
            'Good_depth': [0, 0, 0, 0],  # ignored
        },
    )
    reference = pd.Series(['T', 'T', 'T', 'T'])

    prob_mutation = nucleotide_mutation_prob(
        cell_counts,
        reference,
        error_rate_when_no_mutation=0.007,
        error_rate_when_mutation=0.008,
        p_mutation=1 / 331.0,
    )

    expected = pd.DataFrame(
        data={
            '#CHR': ['MT', 'MT', 'MT', 'MT'],
            'POS': [1, 2, 3, 4],
            'Prob_mutation_A': [1.0, 0.0, 0.0, 0.0],
            'Prob_mutation_C': [0.0, 1.0, 0.0, 0.0],
            'Prob_mutation_G': [0.0, 0.0, 1.0, 0.0],
            'Prob_mutation_T': [0.0, 0.0, 0.0, 1.0],
        }
    )
    assert_frame_equal(expected, prob_mutation)


def test_nucleotide_mutation_prob():
    # Generic counts, changing reference

    cell_counts = pd.DataFrame(
        data={
            '#CHR': ['MT', 'MT'],
            'POS': [1, 2],
            'Count_A': [10, 10],
            'Count_C': [20, 20],
            'Count_G': [ 5,  5],  # noqa
            'Count_T': [ 2,  2],  # noqa
            'Good_depth': [0, 0],  # ignored
        },
    )
    reference = pd.Series(['T', 'C'])

    mutation_prob = nucleotide_mutation_prob(
        cell_counts,
        reference,
        error_rate_when_no_mutation=0.02,
        error_rate_when_mutation=0.01,
        p_mutation=1 / 250.0,
    )

    # Probabilities sum to 1
    assert_almost_equal(mutation_prob[PROB_COLUMNS].sum(axis=1).values, 1.0)

    # Computed by hand
    expected = pd.DataFrame(
        data={
            '#CHR': ['MT', 'MT'],
            'POS': [1, 2],
            'Prob_mutation_A': [4.45915671e-22, 9.99999997e-01],
            'Prob_mutation_C': [1.00000000e+00, 2.75922958e-10],
            'Prob_mutation_G': [1.11355052e-30, 2.49722222e-09],
            'Prob_mutation_T': [2.62000411e-29, 2.04303992e-13],
        }
    )
    assert_frame_equal(expected, mutation_prob, check_like=True)


def test_nucleotide_mutation_prob_row_with_all_zeros():
    # Generic counts, changing reference

    cell_counts = pd.DataFrame(
        data={
            '#CHR': ['MT'],
            'POS': [1],
            'Count_A': [0],
            'Count_C': [0],
            'Count_G': [0],
            'Count_T': [0],
            'Good_depth': [0],
        },
    )
    reference = pd.Series(['T'])

    mutation_prob = nucleotide_mutation_prob(
        cell_counts,
        reference,
        error_rate_when_no_mutation=0.02,
        error_rate_when_mutation=0.01,
        p_mutation=1 / 250.0,
    )

    expected = pd.DataFrame(
        data={
            '#CHR': ['MT'],
            'POS': [1],
            'Prob_mutation_A': [np.nan],
            'Prob_mutation_C': [np.nan],
            'Prob_mutation_G': [np.nan],
            'Prob_mutation_T': [np.nan],
        }
    )
    assert_frame_equal(expected, mutation_prob, check_like=True)


def test_nucleotide_mutation_prob_row_reference_is_N():
    # Generic counts, changing reference

    cell_counts = pd.DataFrame(
        data={
            '#CHR': ['MT'],
            'POS': [1],
            'Count_A': [10],
            'Count_C': [40],
            'Count_G': [7],
            'Count_T': [12],
            'Good_depth': [0],  # ignored
        },
    )
    reference = pd.Series(['N'])

    mutation_prob = nucleotide_mutation_prob(
        cell_counts,
        reference,
        error_rate_when_no_mutation=0.02,
        error_rate_when_mutation=0.01,
        p_mutation=1 / 250.0,
    )

    expected = pd.DataFrame(
        data={
            '#CHR': ['MT'],
            'POS': [1],
            'Prob_mutation_A': [np.nan],
            'Prob_mutation_C': [np.nan],
            'Prob_mutation_G': [np.nan],
            'Prob_mutation_T': [np.nan],
        }
    )
    assert_frame_equal(expected, mutation_prob, check_like=True)


def test_nucleotide_mutation_prob_multiple_references():
    # The nucleotide_mutation_prob function computes the probability by
    # reference type. Makes sure that parts works fine.

    cell_counts = pd.DataFrame(
        data={
            '#CHR': ['MT', 'MT', 'MT', 'MT', 'MT', 'MT', 'MT', 'MT'],
            'POS': [1, 2, 3, 4, 5, 6, 7, 8],
            'Count_A': [0, 12, 10, 43, 10, 11,  0, 70],
            'Count_C': [0, 20, 30,  9, 20, 22, 20, 20],
            'Count_G': [0, 15, 77, 90,  5,  1,  0, 75],  # noqa
            'Count_T': [0, 22, 44, 21,  2,  0,  0, 72],  # noqa
            'Good_depth': [0, 0, 0, 0, 0, 0, 0, 0],  # ignored
        },
    )
    reference = pd.Series(['T', 'C', 'T', 'N', 'A', 'A', 'G', 'A'])

    mutation_prob = nucleotide_mutation_prob(
        cell_counts,
        reference,
        error_rate_when_no_mutation=0.02,
        error_rate_when_mutation=0.01,
        p_mutation=1 / 250.0,
    )

    for idx in range(cell_counts.shape[0]):
        row = cell_counts.iloc[[idx], :]
        row_ref = reference.iloc[[idx]]
        row_mutation_prob = nucleotide_mutation_prob(
            row,
            row_ref,
            error_rate_when_no_mutation=0.02,
            error_rate_when_mutation=0.01,
            p_mutation=1 / 250.0,
        )
        assert_frame_equal(row_mutation_prob,
                           mutation_prob.iloc[[idx], :], check_like=True)

def test_mutation_prob():
    # Generic counts, changing reference

    nucleotide_prob = pd.DataFrame(
        data={
            '#CHR': ['MT', 'MT', 'MT'],
            'POS': [1, 2, 3],
            'Prob_mutation_A': [0.1, 0.1, 0.1],
            'Prob_mutation_C': [0.3, 0.3, 0.3],
            'Prob_mutation_G': [0.2, 0.2, 0.2],
            'Prob_mutation_T': [0.4, 0.4, 0.4],
        }
    )
    reference = pd.Series(['A', 'G', 'N'])
    p_mutation = mutation_prob(nucleotide_prob, reference)

    # Computed by hand
    expected = pd.DataFrame(
        data={
            '#CHR': ['MT', 'MT', 'MT'],
            'POS': [1, 2, 3],
            'Prob_mutation': [0.9, 0.8, np.nan],
        }
    )
    assert_frame_equal(expected, p_mutation, check_like=True)
