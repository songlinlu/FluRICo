# Install and load packages
if(!require("WGCNA")) {
  install.packages("WGCNA")
  library(WGCNA)
}
if(!require("impute")) {
  if(!require("BiocManager")) install.packages("BiocManager")
  BiocManager::install("impute")
  library(impute)
}
if(!require("igraph")) install.packages("igraph")
if(!require("svglite")) install.packages("svglite") # Export dependency
library(igraph)
library(svglite)

result_dir <- "WGCNA_all_sample_Results"
if(!dir.exists(result_dir)) dir.create(result_dir)

options(stringsAsFactors = FALSE)
enableWGCNAThreads()

# ================= Data Preparation =================
cat("\n========== Data Preparation ==========\n")
df_data <- as.data.frame(all_ko_mt_clr)
sample_ids <- df_data[, 1]
ko_data <- df_data[, -1]
rownames(ko_data) <- sample_ids

if(sum(is.na(ko_data)) > 0) {
  imputed_obj <- impute.knn(as.matrix(t(ko_data)))
  ko_data <- as.data.frame(t(imputed_obj$data))
  rownames(ko_data) <- sample_ids
}

datExpr <- ko_data
disease_labels <- gsub("[0-9]+$", "", rownames(datExpr))

correct_order <- c("HC", "PL", "LC")
disease_factor <- factor(disease_labels, levels = correct_order)

clinical_data <- data.frame(
  Disease_Status = as.numeric(disease_factor),  # HC=1, PL=2, LC=3
  row.names = rownames(datExpr)
)

gsg <- goodSamplesGenes(datExpr, verbose = 3)
if(!gsg$allOK) {
  datExpr <- datExpr[gsg$goodSamples, gsg$goodGenes]
  clinical_data <- clinical_data[rownames(datExpr), , drop=FALSE]
}

cat(paste("Number of samples:", nrow(datExpr), "\n"))
cat(paste("Number of KOs:", ncol(datExpr), "\n"))

# ================= Soft Threshold Selection =================
cat("\n========== Soft Threshold Selection ==========\n")
powers <- c(1:20)
sft <- pickSoftThreshold(datExpr, powerVector = powers, verbose = 5)

if(!is.na(sft$powerEstimate)) {
  power <- sft$powerEstimate
} else {
  r2_values <- -sign(sft$fitIndices[,3])*sft$fitIndices[,2]
  power_indices <- which(r2_values > 0.8)
  power <- ifelse(length(power_indices) > 0, powers[min(power_indices)], 6)
}
cat(paste("Selected soft threshold:", power, "\n"))

# Save as SVG
svglite(file.path(result_dir, "1_soft_threshold.svg"), width = 10, height = 5)
par(mfrow = c(1,2))
plot(sft$fitIndices[,1], -sign(sft$fitIndices[,3])*sft$fitIndices[,2],
     xlab="Soft Threshold (power)",
     ylab="Scale Free Topology Model Fit, signed R^2",
     main="Scale independence", type="n")
text(sft$fitIndices[,1], -sign(sft$fitIndices[,3])*sft$fitIndices[,2], 
     labels=powers, cex=0.9, col="red")
abline(h=0.80, col="red")
plot(sft$fitIndices[,1], sft$fitIndices[,5],
     xlab="Soft Threshold (power)",
     ylab="Mean Connectivity", 
     main="Mean connectivity", type="n")
text(sft$fitIndices[,1], sft$fitIndices[,5], 
     labels=powers, cex=0.9, col="red")
dev.off()

# ================= Network Construction and KO Clustering =================
cat("\n========== KO Clustering Analysis (Core Step) ==========\n")
cat("Computing KO similarity and performing clustering...\n")

net <- blockwiseModules(
  datExpr,
  power = power,
  TOMType = "signed",
  #corType = "bicor",
  networkType = "signed hybrid",
  minModuleSize = 40,
  reassignThreshold = 0,
  mergeCutHeight = 0.01,
  numericLabels = TRUE,
  pamRespectsDendro = FALSE,
  pamStage = TRUE,
  saveTOMs = TRUE,
  maxPamDist = 0.3,
  saveTOMFileBase = file.path(result_dir, "KO_TOM"),
  verbose = 3,
  deepSplit = 4
)

moduleColors <- labels2colors(net$colors)
cat("\n========== KO Clustering Results ==========\n")
cat(paste("Clustered", ncol(datExpr), "KOs into", length(unique(moduleColors)), "functional modules\n\n"))
print(sort(table(moduleColors), decreasing = TRUE))

# ================= Visualization 1: KO Clustering Dendrogram =================
# Save as SVG
svglite(file.path(result_dir, "2_KO_clustering_dendrogram.svg"), width = 14, height = 8)
plotDendroAndColors(
  net$dendrograms[[1]],
  moduleColors[net$blockGenes[[1]]],
  "Modules",
  dendroLabels = FALSE,
  hang = 0.03,
  addGuide = TRUE,
  guideHang = 0.05,
  main = "KO Clustering Dendrogram and Module Assignment"
)
dev.off()

# Save KO module assignments
ko_modules <- data.frame(
  KO = colnames(datExpr),
  Module = moduleColors,
  stringsAsFactors = FALSE
)
write.csv(ko_modules, file.path(result_dir, "KO_module_assignment.csv"), row.names = FALSE)

# ================= Visualization 2: Module-Disease Relationships (Spearman) =================
cat("\n========== Analyzing Module-Disease Relationships (Spearman) ==========\n")

MEs <- moduleEigengenes(datExpr, moduleColors)$eigengenes
MEs <- orderMEs(MEs)

moduleTraitCor <- cor(MEs, clinical_data$Disease_Status, method = "spearman")

moduleTraitPvalue <- apply(MEs, 2, function(me) {
  cor.test(me, clinical_data$Disease_Status, method = "spearman")$p.value
})
moduleTraitPvalue <- matrix(moduleTraitPvalue,
                            nrow = nrow(moduleTraitCor),
                            dimnames = list(rownames(moduleTraitCor), "Disease_Status"))

moduleTraitPvalue_FDR <- matrix(
  p.adjust(moduleTraitPvalue, method = "fdr"),
  nrow = nrow(moduleTraitPvalue),
  dimnames = list(rownames(moduleTraitPvalue), "Disease_Status")
)

cor_results <- data.frame(
  Module = gsub("ME", "", rownames(moduleTraitCor)),
  KO_count = sapply(gsub("ME", "", rownames(moduleTraitCor)),
                    function(x) sum(moduleColors == x)),
  Correlation = moduleTraitCor[,1],
  Pvalue = moduleTraitPvalue[,1],
  FDR = moduleTraitPvalue_FDR[,1],
  Significant = ifelse(moduleTraitPvalue_FDR[,1] < 0.05, "Yes", "No"),
  stringsAsFactors = FALSE
)
cor_results <- cor_results[order(abs(cor_results$Correlation), decreasing = TRUE),]
write.csv(cor_results, file.path(result_dir, "module_disease_correlation.csv"), row.names = FALSE)

# Save as SVG
svglite(file.path(result_dir, "3_module_disease_heatmap.svg"), width = 8, height = 10)
textMatrix <- paste0(
  signif(moduleTraitCor, 2),
  "\n(P=", signif(moduleTraitPvalue, 3),
  ")\n(FDR=", signif(moduleTraitPvalue_FDR, 3), ")"
)
dim(textMatrix) <- dim(moduleTraitCor)

labeledHeatmap(
  Matrix = moduleTraitCor,
  xLabels = "Disease Status",
  yLabels = names(MEs),
  ySymbols = names(MEs),
  colorLabels = FALSE,
  colors = blueWhiteRed(50),
  textMatrix = textMatrix,
  setStdMargins = FALSE,
  cex.text = 0.8,
  zlim = c(-1,1),
  main = "Module-Disease Relationships (Spearman)"
)
dev.off()

# ================= Visualization 3: Per-Module KO Lists (KS via Spearman) =================
cat("\n========== Exporting Detailed KO Lists per Module (KS = Spearman) ==========\n")

all_colors <- unique(moduleColors)
all_colors <- all_colors[all_colors != "grey"]

for(color in all_colors) {
  module_kos <- colnames(datExpr)[moduleColors == color]
  module_eigengene <- MEs[, paste0("ME", color)]
  
  module_MM <- cor(datExpr[, module_kos], module_eigengene)
  module_MM.p <- corPvalueStudent(module_MM, nSamples = nrow(datExpr))
  
  module_KS <- apply(datExpr[, module_kos, drop=FALSE], 2, function(x) {
    cor(x, clinical_data$Disease_Status, method = "spearman")
  })
  
  module_KS.p <- apply(datExpr[, module_kos, drop=FALSE], 2, function(x) {
    cor.test(x, clinical_data$Disease_Status, method = "spearman")$p.value
  })
  
  module_MM.fdr <- p.adjust(module_MM.p, method = "fdr")
  module_KS.fdr <- p.adjust(module_KS.p, method = "fdr")
  
  module_info <- data.frame(
    KO = module_kos,
    ModuleMembership = module_MM[,1], 
    MM.pvalue = module_MM.p[,1],
    MM.FDR = module_MM.fdr,
    DiseaseCorrelation = module_KS,   
    Disease.pvalue = module_KS.p,     
    Disease.FDR = module_KS.fdr,
    stringsAsFactors = FALSE
  )
  
  module_info <- module_info[order(abs(module_info$ModuleMembership), decreasing = TRUE), ]
  write.csv(module_info, file.path(result_dir, paste0("Module_", color, "_KOs.csv")), row.names = FALSE)
}

cat("\n========== KO Clustering Analysis Complete (Spearman Version) ==========\n")
cat("All results saved in:", result_dir, "\n")
