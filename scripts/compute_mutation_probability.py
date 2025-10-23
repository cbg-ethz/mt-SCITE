import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

from functions import compute_mutation_prob_per_cell, compute_mutation_prob_matrix
#Note that within functions, there is also a source statement, that reads the functions for the mutation probability calculation into memory. These can be found under mt-SCITE/Notebooks/mito/genotyping.py


def main(input_file, output_file, reference_file, error_rate, prior_mutation_prob):

    # input checks
    if not (0 <= error_rate <= 1):
        raise ValueError(f"error_rate must be between 0 and 1, got {error_rate}")
    
    if not (0 <= prior_mutation_prob <= 1):
        raise ValueError(f"prior_mutation_prob must be between 0 and 1, got {prior_mutation_prob}")

    # --- Check reference input ---
    if isinstance(reference_file, (str, os.PathLike)):
        # Treat as a file path
        if not os.path.exists(reference_file):
            raise FileNotFoundError(f"Reference file not found: {reference}")

        try:
            ref_df = pd.read_table(reference_file, delimiter=",")
        except Exception as e:
            raise ValueError(f"Could not read reference file {reference}: {e}")

        # Expect exactly one column of bases
        if ref_df.shape[1] != 1:
            raise ValueError(
                f"Reference file {reference} must contain exactly one column with reference bases, "
                f"but has {ref_df.shape[1]} columns."
            )

    else:
        raise TypeError(
            "reference must be a path to a CSV file."
        )

    # Extract as Series
    reference = pd.Series(ref_df["x"])
    print(reference)
    print(type(reference))
    #ref_series = pd.Series(re
    
    
    # Load the 3D numpy array from the .npy file (cells x mutations x states)
    read_counts_array = np.load(input_file)
    
    # Check array dimensions
    num_cells, num_mutations, num_states = read_counts_array.shape
    if num_states != 4:
        raise ValueError("Each mutation must have 4 states (Count_A, Count_C, Count_G, Count_T).")

    # --- Length check ---
    if ref_df.shape[0] != num_mutations:
        raise ValueError(
            f"The number of bases in the reference ({ref_df.shape[0]}) does not match number of mutations in the input",
            f"({num_mutations})."
        )

    # Convert the 3D numpy array into a list of DataFrames
    read_count_list = []
    for i in range(num_cells):
        # Create a DataFrame for each cell's read counts with required columns
        cell_df = pd.DataFrame(read_counts_array[i], columns=['Count_A', 'Count_C', 'Count_G', 'Count_T'])
        cell_df['#CHR'] = 'chr1'  # Placeholder chromosome identifier
        cell_df['POS'] = range(num_mutations)  # Mutation positions - are essentially ignored
        read_count_list.append(cell_df)


    # Call the compute_mutation_prob_matrix function
    mutation_prob_matrix = compute_mutation_prob_matrix(read_count_list, reference, error_rate, prior_mutation_prob)

    # Drop the #CHR and POS columns, keeping only numerical values for mutation probabilities
    mutation_prob_values = mutation_prob_matrix.iloc[:, 2:]  # Skip the first two columns (#CHR, POS)

    # Save the matrix to the output file without headers or index
    mutation_prob_values.to_csv(output_file, sep=' ', index=False, header=False)

if __name__ == "__main__":
    # Command-line argument parsing
    if len(sys.argv) != 7:
        print("Usage: python compute_mutation_probability.py <input_file> <output_file> <reference_file> <error_rate> <prior_mutation_prob>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    reference_file = sys.argv[3]
    try:
        error_rate = float(sys.argv[4])
    except ValueError:
        print("Error: error_rate must be a float.")
        sys.exit(1)

    try:
        prior_mutation_prob = float(sys.argv[5])
    except ValueError:
        print("Error: Prior mutation prob must be a float.")
        sys.exit(1)

        
    # Run the main function
    main(input_file, output_file, reference_file, error_rate, prior_mutation_prob)
