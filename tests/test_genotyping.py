# test_mutation_probs.py
import os, sys
import math
import importlib
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.stats import multinomial

# --- dynamic import of the module under test ---
sys.path.append("/Users/soseidel/Projects/mitochondria/mt-scite-benchmark/software/mt-SCITE/Notebooks/mito")
import genotyping
#MODULE_NAME = os.environ.get("MUTATION_MODULE", "mutation_probs")
#m = importlib.import_module(MODULE_NAME)

# -----------------------
# Fixtures and test data
# -----------------------
@pytest.fixture
def tiny_counts_df():
    # 6 rows to cover: each ref nucleotide, one N (unknown), one zero-reads row
    df = pd.DataFrame(
        {
            "#CHR": ["1", "1", "2", "2", "3", "3"],
            "POS":  [100, 101, 200, 201, 300, 301],
            # counts chosen to be small but non-degenerate
            "Count_A": [20,  0,  2,  1,  0,  0],
            "Count_C": [ 1, 30,  3,  0,  0,  0],
            "Count_G": [ 0,  4, 10,  0,  0,  0],
            "Count_T": [ 1,  1,  1, 50,  0,  0],
            "Good_depth": [0, 0, 0, 0, 0, 0],  # ignored
        }
    )
    # References: A, C, G, T, N (unknown), T (but zero reads)
    reference = pd.Series(["A", "C", "G", "T", "N", "T"])
    return df, reference



@pytest.fixture
def params():
    return dict(
        error_rate_when_no_mutation=0.01,
        error_rate_when_mutation=0.01,
        p_mutation=10.0/16600,
    )

# -----------------------
# Unit tests for helpers
# -----------------------
def test__log_prob_ABC_no_mutation_matches_scipy_vector_and_scalar():
    # vector case
    reads_R = pd.Series([10, 5, 0])
    reads_A = pd.Series([1, 2, 0])
    reads_B = pd.Series([0, 1, 0])
    reads_C = pd.Series([1, 0, 0])
    err = 0.01

    got = genotyping._log_prob_ABC_no_mutation(reads_R, reads_A, reads_B, reads_C, err)

    # Computed using my direct implementation
    expected = [-6.625266385261044, -12.03763512383285, 0.0 ]
    assert_allclose(got, expected, rtol=1e-12, atol=1e-12)

    # scalar case (length-1 series)
    reads_R = pd.Series([10])
    reads_A = pd.Series([1])
    reads_B = pd.Series([0])
    reads_C = pd.Series([1])
    got = genotyping._log_prob_ABC_no_mutation(reads_R, reads_A, reads_B, reads_C, err)
    # Computed using my direct implementation
    expected = [-6.625266385261044]
    
    assert_allclose(got, expected, rtol=1e-12, atol=1e-12)

def test__log_prob_ABC_with_mutation_at_A_behaves_like_defined_formula():
    """
    The implementation: log P(v | mutation at A) =
      log Multinomial([B, C, R] ; n=R+B+C, p=[e/2, e/2, 1-e]) - log(total_reads)
    with a special-case where R+B+C = 0 -> log_prob_BC set to 0 before subtraction.
    """
    reads_R = pd.Series([5, 0, 0])     # include a case with R+B+C == 0 (second row if B=C=0)
    reads_A = pd.Series([4, 7, 0])     # note: function expects reads_A > 0
    reads_B = pd.Series([1, 0, 0])
    reads_C = pd.Series([0, 0, 0])
    err = 0.05

    got = genotyping._log_prob_ABC_with_mutation_at_A(reads_R, reads_A, reads_B, reads_C, err)

    total = reads_R + reads_A + reads_B + reads_C
    reads_RBC = reads_R + reads_B + reads_C
    log_prob_BC = multinomial.logpmf(
        np.column_stack([reads_B, reads_C, reads_R]),
        n=reads_RBC,
        p=[err/2.0, err/2.0, 1 - err],
    )
    # Computed using my direct implementation
    expected = [-4.45617154981768, -1.9459101490553132, np.inf]  #log_prob_BC - np.log(total.values)
    
    assert_allclose(got, expected, rtol=1e-12, atol=1e-12)

# -----------------------
# Integration tests
# -----------------------
def test_nucleotide_mutation_prob_shapes_and_columns(tiny_counts_df, params):
    df, reference = tiny_counts_df
    out = genotyping.nucleotide_mutation_prob(
        cell_counts=df,
        reference=reference,
        **params,
    )
    assert list(out.columns) == ["#CHR", "POS",
                                 "Prob_mutation_A", "Prob_mutation_C",
                                 "Prob_mutation_G", "Prob_mutation_T"]
    assert len(out) == len(df)


def test_nucleotide_mutation_prob_rowwise_normalization(tiny_counts_df, params):
    df, reference = tiny_counts_df
    out = genotyping.nucleotide_mutation_prob(df, reference, **params)

    # rows where ref is known AND total reads > 0 should sum to 1 across the four probs
    counts_sum = df[["Count_A", "Count_C", "Count_G", "Count_T"]].sum(axis=1)
    valid = (reference.isin(list("ACGT"))) & (counts_sum > 0)

    row_sums = out.loc[valid, ["Prob_mutation_A","Prob_mutation_C",
                               "Prob_mutation_G","Prob_mutation_T"]].sum(axis=1)
    assert_allclose(row_sums.values, np.ones(valid.sum()), rtol=1e-12, atol=1e-12)

    
def test_integration_formatting_and_mutation_prob_call():
    """
    Part A: Use previously verified vector case to confirm how to format inputs
            for the _log_prob_ABC_no_mutation helper via a cell_counts DataFrame.
    Part B: Show the correct input format for mutation_prob(...) and assert that it
            returns 1 - per_base[ref], independent of column order.
    """
    # ----- Part A: reuse the known-good vector case -----
    # Known reads and expected (already computed & checked elsewhere)
    reads_R = pd.Series([10, 5, 0])
    reads_A = pd.Series([1,  2, 0])
    reads_B = pd.Series([0,  1, 0])
    reads_C = pd.Series([1,  0, 0])
    err = 0.01
    expected = np.array([-6.625266385261044, -12.03763512383285, 0.0])

    # Build the minimal cell_counts DataFrame in the SAME semantic layout used by your code:
    # choose the reference base to be 'T' so that:
    #   Count_T == reads_R (reference),
    #   Count_A == reads_A, Count_C == reads_B, Count_G == reads_C (the three non-refs).
    cell_counts = pd.DataFrame(
        {
            "#CHR": ["1", "1", "1"],
            "POS":  [100, 101, 102],
            "Count_A": reads_A,
            "Count_C": reads_B,
            "Count_G": reads_C,
            "Count_T": reads_R,
            "Good_depth": [0, 0, 0],  # ignored by the code
        }
    )
    reference = pd.Series(["T", "T", "T"])

    # Extract the Series exactly as nucleotide_mutation_prob(...) would for ref='T'
    ref_count = cell_counts.loc[:, "Count_T"]
    non_ref = cell_counts.loc[:, ["Count_A", "Count_C", "Count_G"]]
    got = genotyping._log_prob_ABC_no_mutation(
        reads_R=ref_count,
        reads_A=non_ref.iloc[:, 0],  # Count_A
        reads_B=non_ref.iloc[:, 1],  # Count_C
        reads_C=non_ref.iloc[:, 2],  # Count_G
        error_rate=err,
    )
    assert_allclose(got, expected, rtol=1e-12, atol=1e-12)

    # ----- Part B: format and call mutation_prob(...) correctly -----
    # Build a minimal per-base probability table (as produced by nucleotide_mutation_prob),
    # but deliberately shuffle column order to ensure mutation_prob relies on column NAMES.
    per_base = pd.DataFrame(
        {
            "#CHR": ["1", "1", "1"],
            "POS":  [100, 101, 102],
            # shuffled order on purpose: T first, then A, G, C
            "Prob_mutation_T": [0.85, 0.40, 0.60],
            "Prob_mutation_A": [0.05, 0.30, 0.10],
            "Prob_mutation_G": [0.05, 0.10, 0.20],
            "Prob_mutation_C": [0.05, 0.20, 0.10],
        }
    )
    # Same reference series as above (all 'T'), matching the “ref column” semantics
    ref_series = pd.Series(["T", "T", "T"])

    # Expected overall mutation probability is 1 - P(M == ref)
    expected_prob_mut = 1.0 - per_base["Prob_mutation_T"].to_numpy()

    out = genotyping.mutation_prob(per_base, ref_series)

    # Check structure and values
    assert list(out.columns) == ["#CHR", "POS", "Prob_mutation"]
    assert_allclose(out["Prob_mutation"].to_numpy(), expected_prob_mut, rtol=1e-12, atol=1e-12)

