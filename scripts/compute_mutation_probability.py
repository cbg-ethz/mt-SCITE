import numpy as np
import pandas as pd
import sys
from pathlib import Path

from functions import compute_mutation_prob_per_cell, compute_mutation_prob_matrix
#Note that within functions, there is also a source statement, that reads the functions for the mutation probability calculation into memory. These can be found under mt-SCITE/Notebooks/mito/genotyping.py


def main(input_file, output_file, error_rate):
    # Load the 3D numpy array from the .npy file (cells x mutations x states)
    read_counts_array = np.load(input_file)
    
    # Check array dimensions
    num_cells, num_mutations, num_states = read_counts_array.shape
    if num_states != 4:
        raise ValueError("Each mutation must have 4 states (Count_A, Count_C, Count_G, Count_T).")

    # Convert the 3D numpy array into a list of DataFrames
    read_count_list = []
    for i in range(num_cells):
        # Create a DataFrame for each cell's read counts with required columns
        cell_df = pd.DataFrame(read_counts_array[i], columns=['Count_A', 'Count_C', 'Count_G', 'Count_T'])
        cell_df['#CHR'] = 'chr1'  # Placeholder chromosome identifier
        cell_df['POS'] = range(1, num_mutations + 1)  # Mutation positions starting from 1
        read_count_list.append(cell_df)

    # Create a reference Series of "A" with the same length as the number of mutations
    reference = pd.Series(['A'] * num_mutations, index=range(num_mutations))

    # Call the compute_mutation_prob_matrix function
    mutation_prob_matrix = compute_mutation_prob_matrix(read_count_list, reference, error_rate)

    # Drop the #CHR and POS columns, keeping only numerical values for mutation probabilities
    mutation_prob_values = mutation_prob_matrix.iloc[:, 2:]  # Skip the first two columns (#CHR, POS)

    # Save the matrix to the output file without headers or index
    mutation_prob_values.to_csv(output_file, sep=' ', index=False, header=False)

if __name__ == "__main__":
    # Command-line argument parsing
    if len(sys.argv) != 4:
        print("Usage: python compute_mutation_probability.py <input_file> <output_file> <error_rate>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    try:
        error_rate = float(sys.argv[3])
    except ValueError:
        print("Error: error_rate must be a float.")
        sys.exit(1)
    
    # Run the main function
    main(input_file, output_file, error_rate)
