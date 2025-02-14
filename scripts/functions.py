import os

## functions to compute mutation probability matrix for mt-scite
# import joanna's mutation probability calculation
script_dir = os.path.dirname(__file__)
calculations_path = os.path.join(script_dir, '../Notebooks/mito/genotyping.py')

with open(calculations_path) as f:
    exec(f.read())

# functions that wrap the calculations, based on joanna's cell_prob(e_rate)

def compute_mutation_prob_per_cell(read_count, reference, error_rate, prior_mutation_prob=0.001):

    nucleotide_probabilities = nucleotide_mutation_prob(
            cell_counts = read_count,
            reference = reference,
            error_rate_when_no_mutation = error_rate, ###
            error_rate_when_mutation = error_rate, ###
            p_mutation = prior_mutation_prob,
        )
    mutation_probabilities = mutation_prob(nucleotide_probabilities, reference)

    return mutation_probabilities

    

def compute_mutation_prob_matrix(read_count_list, reference, error_rate, prior_mutation_prob=0.001):
    """
    Computes the mutation probability matrix for given cell counts and reference.
    
    Parameters
    ----------
    read_count_list : list of pd.DataFrame
        A list where each element is a DataFrame containing read counts for a single cell. Each DataFrame in the list 
        must have the following structure:
            - Required columns: '#CHR', 'POS', 'Count_A', 'Count_C', 'Count_G', 'Count_T'
            - Each row corresponds to a unique genomic position.
            - All DataFrames in the list must cover the same genomic positions in the same order.
    
    reference : pd.Series
        Series where each index represents a genomic position, and each value specifies the reference nucleotide 
        (e.g., 'A', 'C', 'G', 'T', or 'N' for unknown) at that position. The series must match the positions
        in each DataFrame in `read_count_list`.
    
    error_rate : float
        The error rate applied in calculations. Must be a value between 0 and 1.
    
    prior_mutation_prob : float, optional
        Prior probability of mutation at each position. Defaults to 0.001. Must be a value between 0 and 1.
    
    Returns
    -------
    pd.DataFrame
        A mutation probability matrix DataFrame, with columns:
            - '#CHR': Chromosome identifier for each position (from `read_count_list[0]`).
            - 'POS': Position within the chromosome (from `read_count_list[0]`).
            - One column per cell, labeled 'Cell_1', 'Cell_2', etc., containing the mutation probability
              for each position in each cell.
    
    Requirements
    ------------
    - All DataFrames in `read_count_list` must have the same structure and cover the same genomic positions.
    - Ensure that `reference` is aligned with the positions in `read_count_list`.
    - `error_rate` and `prior_mutation_prob` must be valid probabilities between 0 and 1.
    """

    # Initialize the mutation probability matrix using #CHR and POS columns from the first read_count DataFrame
    mutation_prob_matrix = read_count_list[0][['#CHR', 'POS']].copy()

    # Compute mutation probabilities for each cell (col in mutation prob matrix) at each site (row)  
    for i, read_count in enumerate(read_count_list):
         
         cell_mutation_probs = compute_mutation_prob_per_cell(read_count, reference, error_rate, prior_mutation_prob)

         # Add the computed mutation probabilities as a new column in the matrix
         mutation_prob_matrix[f'Cell_{i+1}'] = cell_mutation_probs['Prob_mutation'].values

    return mutation_prob_matrix


