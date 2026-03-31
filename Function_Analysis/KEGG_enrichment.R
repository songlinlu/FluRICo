suppressPackageStartupMessages({
  library(clusterProfiler)
  library(enrichplot)
  library(ggplot2)
})

## ========= 1. Configuration (modify variables here) =========

# Target data frame variable
target_df_var <- module_turquoise

# Background (universe) data frame variable
universe_df_var <- allko

# Column name containing KO IDs in the background data frame
universe_col_name <- "0"  

# Task name (used as output folder name)
my_task_name <- "module_turquoise_ko_enrich"


## ========= 2. Helper Functions =========

# Extract KO IDs from a data frame
extract_ids <- function(df, col_name = NULL) {
  df <- as.data.frame(df)
  if (!is.null(col_name) && col_name %in% names(df)) {
    ids <- as.character(df[[col_name]])
  } else {
    ids <- as.character(df[, 1])
  }
  ids <- ids[!is.na(ids) & nzchar(ids)]
  unique(ids)
}

# Remove possible "ko:" prefix from KEGG IDs
clean_kegg_id <- function(id_vec) {
  unique(sub("^ko:", "", as.character(id_vec)))
}

## ========= 3. Data Loading and Validation =========

message("--------------------------------------------------")
message(">>> Extracting data...")

# 1. Process target KO list
target_ko <- extract_ids(target_df_var)
target_ko <- clean_kegg_id(target_ko)

# 2. Process universe KO list
universe_ko <- extract_ids(universe_df_var, col_name = universe_col_name)
universe_ko <- clean_kegg_id(universe_ko)

# 3. Summary statistics
message(sprintf("Target KO count   : %d", length(target_ko)))
message(sprintf("Universe KO count  : %d", length(universe_ko)))

# 4. Check overlap between target and universe
common_ids <- intersect(target_ko, universe_ko)
message(sprintf("Overlapping KOs    : %d (used for enrichment)", length(common_ids)))

if (length(common_ids) == 0) {
  stop("Error: No overlapping IDs between target and universe. Please check variable names.")
} else {
  message("Data validation passed. ID types are consistent (K numbers).")
}

## ========= 4. Enrichment Analysis =========

run_kegg_enrichment <- function(target_list, universe_list, task_name, output_dir = "KEGG_Results") {
  
  # Create output directory
  out_path <- file.path(output_dir, task_name)
  if (!dir.exists(out_path)) dir.create(out_path, recursive = TRUE)
  
  message(sprintf("\n>>> Running enrichKEGG: %s ...", task_name))
  
  # Run enrichment
  enrich_res <- tryCatch(
    enrichKEGG(
      gene          = target_list,
      universe      = universe_list,
      organism      = "ko",
      keyType       = "kegg",
      pvalueCutoff  = 0.05,
      qvalueCutoff  = 0.20,
      pAdjustMethod = "BH"
    ),
    error = function(e) { message("Error: ", e$message); return(NULL) }
  )
  
  if (is.null(enrich_res)) return(NULL)
  
  res_df <- as.data.frame(enrich_res)
  
  # Save results and generate plots
  if (nrow(res_df) == 0) {
    message("Analysis complete, but no significantly enriched pathways found (q < 0.2).")
    write.csv(res_df, file.path(out_path, paste0(task_name, "_NO_SIG_RESULTS.csv")), row.names = FALSE)
  } else {
    # Save results table
    csv_file <- file.path(out_path, paste0(task_name, "_results.csv"))
    write.csv(res_df, csv_file, row.names = FALSE)
    message(sprintf("Results table saved: %s", csv_file))
    
    # Print top 5 results
    print(head(res_df[, c("ID", "Description", "p.adjust", "Count")], 5))
    
    message(">>> Generating plots...")
    
    # Plot display parameters
    show_num <- 15    
    wrap_width <- 45  
    
    # 1. Dotplot
    # Convert GeneRatio "a/b" to numeric for automatic x-axis scaling
    geneRatio_to_num <- function(gr) {
      sapply(strsplit(as.character(gr), "/"), function(x) as.numeric(x[1]) / as.numeric(x[2]))
    }
    max_gr <- max(geneRatio_to_num(enrich_res@result$GeneRatio), na.rm = TRUE)
    
    p1 <- dotplot(enrich_res, showCategory = show_num, label_format = wrap_width) +
      scale_size_continuous(range = c(5, 12)) +
      ggtitle(paste0("KEGG: ", task_name)) +
      theme_bw() +
      # Add 8% padding on the right to prevent bubble clipping
      scale_x_continuous(limits = c(0, max_gr * 1.08)) +
      coord_cartesian(clip = "off") +
      theme(
        plot.title = element_text(hjust = 0.5, size = 18, face = "bold"),
        axis.text.y = element_text(size = 14, color = "black"),
        axis.text.x = element_text(size = 14, color = "black"),
        axis.title  = element_text(size = 16, face = "bold"),
        legend.text = element_text(size = 12),
        legend.title = element_text(size = 14),
        plot.margin = margin(10, 35, 10, 10)
      )
    
    ggsave(file.path(out_path, paste0(task_name, "_dotplot.svg")), 
           p1, width = 9, height = 8, device = "svg")
    
    # 2. Barplot
    p2 <- barplot(enrich_res, showCategory = show_num, label_format = wrap_width) + 
      ggtitle(paste0("KEGG: ", task_name)) +
      theme_bw() +
      theme(
        plot.title = element_text(hjust = 0.5, size = 18, face = "bold"),
        axis.text.y = element_text(size = 14, color = "black"),
        axis.text.x = element_text(size = 14, color = "black"),
        axis.title = element_text(size = 16, face = "bold"),
        plot.margin = margin(10, 10, 10, 10)
      )
    
    ggsave(file.path(out_path, paste0(task_name, "_barplot.svg")), 
           p2, width = 9, height = 8, device = "svg")
    
    message("SVG plots saved.")
  }
}

## ========= 5. Execute =========
run_kegg_enrichment(target_ko, universe_ko, my_task_name)
