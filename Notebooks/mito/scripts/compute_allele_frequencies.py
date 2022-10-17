from argparse import ArgumentParser
from glob import glob
import os

import pandas as pd

from mito.genotyping import allele_frequency


def read_cell_counts(filename):
    """ Read cell counts file. """
    counts = pd.read_csv(filename, sep='\t')
    return counts


def write_allele_frequencies(output_filename, af):
    """ Save allele frequency to file. """
    af.to_csv(output_filename, sep='\t', index=False)


def main():
    parser = ArgumentParser(description='Process some integers.')
    parser.add_argument('input_path', type=str,
                        help='input path containing the cell count types')
    parser.add_argument('output_path', type=str,
                        help='output path where the allele frequency files are stored')  # noqa

    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path

    # Get all cell counts files in the input path
    cell_count_filenames = list(glob(os.path.join(input_path, '*.txt')))

    for filename in cell_count_filenames:
        print('Processing {}'.format(filename))
        # Load data
        counts = read_cell_counts(filename)

        # Compute allele frequency
        af = allele_frequency(counts)

        # Output result
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        basename = os.path.basename(filename)
        output_filename = os.path.join(output_path, basename)
        print('Writing results to {}'.format(output_filename))
        write_allele_frequencies(output_filename, af)


if __name__ == '__main__':
    main()
