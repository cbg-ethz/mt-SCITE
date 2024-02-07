# Tree inference from mitochondrial mutations
mt-SCITE infers trees from a matrix of mutation probabilities built from a statistical model of alternate read counts in mitochondrial sequencing reads.

## Installation
```
clang++ src/*.cpp -o mtscite
```

## Usage
```
mtscite -i <mut_probabilities> -n <n_sites> -m <n_cells> -l <n_iters> -seed <random_seed>
```

## End-to-end analysis
We also provide notebooks and scripts to perform an end-to-end tree inference analysis using mt-SCITE.

### Environment
```
conda create -n mtSCITE python=3.10
conda install -c conda-forge biopython
conda install -c anaconda pandas
conda install -c anaconda scipy
conda install -c conda-forge matplotlib
conda install -c anaconda seaborn
conda install -c anaconda graphviz
pip install --global-option=build_ext --global-option="-I~/miniconda3/envs/mtSCITE/include"  --global-option="-L~/miniconda3/envs/mtSCITE/include/lib/" pygraphviz
conda install -c anaconda python-graphviz
conda install -c anaconda pydot
conda install -c anaconda networkx
conda install -c anaconda numpy=1.22
```

# Workflow

Raw sequencing data was processed by the snakemake pipeline in preprocessing_pipeline/Snakefile.
This generated tsv files specifying the number of reads supporting the four different nucleotides A, C, G, T and total number of reads for each position in the mitochondrial genome for each sample.

### Computing mutation probabilities

`compute_mutations_probabilities.ipynb` was run to generate mutation probability matrices for a range of error rates. The mutation probability matrices were stored in `/path/to/matrices/`


### Selecting the error rate
We used a 3-fold cross validation procedure to select the error rate that we used to infer the trees using mt-SCITE. After generating probability matrices for different error rates and storing them in `/path/to/matrices/`, you can perform cross-validation by running this command: 
```
python scripts/cv.py </path/to/matrices/> </path/to/mtscite>
```
This will create a CSV file named `val_scores.csv` with the 3-fold cross-validation likelihood scores. 

`learn_error_rate.ipynb` was run to analyze `val_scores.csv` and to select the error rate to be used for tree generation

### Build trees

Run mt-SCITE with this command:
`</path/to/mtscite> -i pmat.csv -n <n_mutations> -m <n_samples> -r 1 -l 200000 -fd 0.0001 -ad 0.0001 -cc 0.0 -s -a -o </path/to/output/run_id>`

where 
`pmat.csv` is the mutation probability matrix generated with the learned error rate, `n_mutations` is the number of rows in `pmat.csv` and `n_samples` is the number of columns in `pmat.csv`.

