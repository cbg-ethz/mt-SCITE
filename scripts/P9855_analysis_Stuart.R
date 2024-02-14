library(Signac)
library(Seurat)
library(ggplot2)
library(patchwork)
library(EnsDb.Hsapiens.v75)
library(ape)
library(dplyr)
library(stringr)

# Obtain heteroplasmy matrix using Signac following https://stuartlab.org/signac/articles/mito.html
# read the mitochondrial data
#tf1.data <- ReadMGATK(dir = "../../yfv2001/mgatk_yfv2001_ouput/final/")
tf1.data <- ReadMGATK(dir = "/Users/johard/Documents/mgatk_for_mtSCITE_ms/data/mgatk_P9855_output/final/")



# create a Seurat object
tf1 <- CreateSeuratObject(
  counts = tf1.data$counts,
  meta.data = tf1.data$depth,
  assay = "mito"
)

# Call variants
DefaultAssay(tf1) <- "mito"
variants <- IdentifyVariants(tf1, refallele = tf1.data$refallele)

# At least two cells share mutation
high.conf <- subset(
  variants, subset = n_cells_conf_detected >= 2 &
    strand_correlation >= 0.65 &
    vmr > 0.01
)

# Compute allele frequencies
tf1 <- AlleleFreq(tf1, variants = high.conf$variant, assay = "mito")

# Turn to procedure used in https://github.com/caleblareau/mtscATACpaper_reproducibility/blob/master/figures_TF1_GM11906_mixing/code/22_TF1_visualize_variants.R
DefaultAssay(tf1) <- "alleles"
af_filter_mat <- GetAssayData(object = tf1) 

# Get clusters
seuratSNN <- function(matSVD, resolution = 1, k.param = 10){ 
  set.seed(1)
  rownames(matSVD) <- make.unique(rownames(matSVD))
  obj <- FindNeighbors(matSVD, k.param = k.param, annoy.metric = "cosine")
  clusters <- FindClusters(object = obj$snn, resolution = resolution)
  return(as.character(clusters[,1]))
}

# Run clustering on square root heteroplasmies
cl <- seuratSNN(sqrt(t(af_filter_mat)), 1.0, 10)
cl <- str_pad(cl, 2, pad = "0")

names_clusters <- unique(cl)

afp <- af_filter_mat
aftree <- afp
afp[afp < 0.01] <- 0
afp[afp > 0.1] <- 0.1

df <- data.frame(
  cell_id = colnames(afp), 
  cluster_id = as.character(cl)
) %>% arrange(cl)

# Get group means 
matty <- sapply(names_clusters, function(cluster){
  cells <- df %>% dplyr::filter(cluster_id == cluster) %>% pull(cell_id) %>% as.character()
  Matrix::rowMeans(sqrt(afp[,cells]))
})

# Do cosine distance; note that we used sqrt transformation already 
mito.hc <- hclust(dist(lsa::cosine((matty))))

# TODO: compute and save purity data

# Define the file path where you want to save the output
#cluster_file_path <- "../../yfv2001/YFV2001_clusters.txt"
cluster_file_path <- "/Users/johard/Documents/mgatk_for_mtSCITE_ms/output/P9855_clusters.txt"

#tree_file_path <- "../../yfv2001/YFV2001_tree.pdf"
tree_file_path <- "/Users/johard/Documents/mgatk_for_mtSCITE_ms/output/P9855_tree.pdf"



# Write the clusters to a text file
write.table(df, file = cluster_file_path, sep = "\t", col.names = TRUE, row.names = TRUE, quote = FALSE)

# Write the tree to pdf
pdf(tree_file_path, width = 5, height = 5)
plot(mito.hc)
dev.off()

# Print a message indicating the file has been saved
print(paste("Cluster assignments saved at", cluster_file_path))
print(paste("Tree saved at", tree_file_path))

