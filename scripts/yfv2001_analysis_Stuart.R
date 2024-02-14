library(Signac)
library(Seurat)
library(ggplot2)
library(patchwork)
library(EnsDb.Hsapiens.v75)
library(ape)

# read the mitochondrial data
tf1.data <- ReadMGATK(dir = "/Users/johard/Documents/mgatk_for_mtSCITE_ms/data/mgatk_yfv2001_ouput/final/")


# create a Seurat object
tf1 <- CreateSeuratObject(
  counts = tf1.data$counts,
  meta.data = tf1.data$depth,
  assay = "mito"
)


DefaultAssay(tf1) <- "mito"
variants <- IdentifyVariants(tf1, refallele = tf1.data$refallele)


VariantPlot(variants)

# At least two cells share mutation
high.conf <- subset(
  variants, subset = n_cells_conf_detected >= 2 &
    strand_correlation >= 0.65 &
    vmr > 0.01
)


tf1 <- AlleleFreq(tf1, variants = high.conf$variant, assay = "mito")
tf1[["alleles"]]


DefaultAssay(tf1) <- "alleles"
tf1 <- FindClonotypes(tf1)


table(Idents(tf1))



DoHeatmap(tf1, features = VariableFeatures(tf1), slot = "data", disp.max = 0.1) +
  scale_fill_viridis_c()



###### TREE BUILDING SEURAT #######
tf1 <- BuildClusterTree(object = tf1)
PlotClusterTree(object = tf1)



# Save purity data

# Get the output of Idents(tf1)
sample_idents <- Idents(tf1)

# Define the file path where you want to save the output
file_path <- "/Users/johard/Documents/mgatk_for_mtSCITE_ms/YFV2001_clusters.txt"

# Write the output to a text file
write.table(sample_idents, file = file_path, sep = "\t", col.names = TRUE, row.names = TRUE, quote = FALSE)

# Print a message indicating the file has been saved
print(paste("Idents(tf1) saved as", file_path))

