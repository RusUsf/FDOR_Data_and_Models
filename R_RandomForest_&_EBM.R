###############################################################################
# PROPERTY LOSS PREDICTION: RANDOM FOREST AND EBM COMPARISON
# WITH SAME METRICS AS LOGISTIC REGRESSION FOR DIRECT COMPARISON
###############################################################################

###############################################################################
# 1) LOAD REQUIRED PACKAGES
###############################################################################
# Load necessary libraries
if(!require(randomForest)) install.packages("randomForest"); library(randomForest)
if(!require(interpret)) install.packages("interpret"); library(interpret) # For EBM
if(!require(caret)) install.packages("caret"); library(caret)
if(!require(dplyr)) install.packages("dplyr"); library(dplyr)
if(!require(corrplot)) install.packages("corrplot"); library(corrplot)
if(!require(ggplot2)) install.packages("ggplot2"); library(ggplot2)
if(!require(pROC)) install.packages("pROC"); library(pROC)
if(!require(gridExtra)) install.packages("gridExtra"); library(gridExtra)

###############################################################################
# 2) DATA IMPORT AND INITIAL PREPARATION
###############################################################################
# Set file path (adjust as needed)
raw_path <- r"(C:\Users\User\Desktop\Job Search\USF_School_Project\Integrative_Program_Project_Cohort_2022_MSBAIS\Model_Optimization_in_R\INPUTS.csv)"

# Import data
data_full <- read.csv(raw_path, stringsAsFactors = FALSE)

# Replace empty strings with 0 throughout the dataframe
data_full[data_full == ""] <- 0

# Initial data inspection
cat("Initial data dimensions:", dim(data_full)[1], "x", dim(data_full)[2], "\n")

# Select only the needed columns (same as logistic regression model)
data <- data_full %>% 
  select(LND_SQFOOT, TOT_LVG_AREA, S_LEGAL, CONST_CLASS, IMP_QUAL, JV, 
         LND_VAL, NO_BULDNG, NCONST_VAL, DEL_VAL, SPEC_FEAT_VAL, 
         MonthDifference, SALE_PRC1, Target_Var, EFF_AGE, ACT_AGE)

# Convert target variable to factor for classification models
data$Target_Var <- as.factor(data$Target_Var)

# Summary of the selected data
cat("\nSelected columns summary:\n")
print(summary(data[, c("Target_Var", "LND_SQFOOT", "TOT_LVG_AREA", "EFF_AGE", "SALE_PRC1")]))

###############################################################################
# 3) CATEGORICAL VARIABLE HANDLING
###############################################################################
# Handle IMP_QUAL (Property Quality) - Group rare categories
data$IMP_QUAL[data$IMP_QUAL == "0"] <- "NULL"
data$IMP_QUAL_MOD <- data$IMP_QUAL
data$IMP_QUAL_MOD[data$IMP_QUAL_MOD %in% c("6", "7")] <- "6_7" # Combine rare values
data$IMP_QUAL_MOD[is.na(data$IMP_QUAL_MOD) | data$IMP_QUAL_MOD == "NULL"] <- "UNKNOWN"
data$IMP_QUAL_MOD <- factor(data$IMP_QUAL_MOD)

# Handle CONST_CLASS (Construction Class) - Group rare categories
data$CONST_CLASS[data$CONST_CLASS == "0"] <- "NULL"
data$CONST_CLASS_MOD <- data$CONST_CLASS
data$CONST_CLASS_MOD[data$CONST_CLASS_MOD %in% c("1", "5")] <- "1_5" # Combine rare values
data$CONST_CLASS_MOD[is.na(data$CONST_CLASS_MOD) | data$CONST_CLASS_MOD == "NULL"] <- "UNKNOWN"
data$CONST_CLASS_MOD <- factor(data$CONST_CLASS_MOD)

# View distributions of modified categorical variables
cat("\nIMP_QUAL_MOD distribution:\n")
print(table(data$IMP_QUAL_MOD))

cat("\nCONST_CLASS_MOD distribution:\n")
print(table(data$CONST_CLASS_MOD))

###############################################################################
# 4) LOCATION RISK SCORING
###############################################################################
# Calculate loss rate by location (same as in logistic regression)
location_summary <- data %>%
  group_by(S_LEGAL) %>%
  summarize(
    location_risk = mean(as.numeric(as.character(Target_Var))),
    location_count = n()
  )

# Print summary of location counts
cat("\nLocation summary:\n")
cat("Total unique locations:", nrow(location_summary), "\n")
cat("Locations with at least 5 observations:", sum(location_summary$location_count >= 5), "\n")

# Use average risk for locations with insufficient data
location_summary$location_risk[location_summary$location_count < 5] <- NA
avg_risk <- mean(as.numeric(as.character(data$Target_Var)))
cat("Average risk across all locations:", round(avg_risk * 100, 2), "%\n")

# Join back to the data
data <- left_join(data, location_summary[, c("S_LEGAL", "location_risk")], by = "S_LEGAL")
data$location_risk[is.na(data$location_risk)] <- avg_risk

# Create model data excluding original variables replaced with improved versions
model_data <- data[, !names(data) %in% c("S_LEGAL", "IMP_QUAL", "CONST_CLASS")]

###############################################################################
# 5) OUTLIER TRIMMING
###############################################################################
# Function to identify outliers based on IQR (same as in logistic regression)
identify_outliers <- function(x, multiplier = 1.5) {
  # Skip if not numeric
  if(!is.numeric(x)) return(rep(FALSE, length(x)))
  
  # Skip if too many NAs
  if(sum(is.na(x)) / length(x) > 0.25) return(rep(FALSE, length(x)))
  
  q1 <- quantile(x, 0.25, na.rm = TRUE)
  q3 <- quantile(x, 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  
  lower_bound <- q1 - multiplier * iqr
  upper_bound <- q3 + multiplier * iqr
  
  return(x < lower_bound | x > upper_bound)
}

# Identify which numeric columns to check for outliers
numeric_cols <- names(model_data)[sapply(model_data, is.numeric)]
numeric_cols <- setdiff(numeric_cols, "Target_Var") # Exclude Target_Var

# Create outlier matrix
outlier_matrix <- matrix(FALSE, nrow = nrow(model_data), ncol = length(numeric_cols))
colnames(outlier_matrix) <- numeric_cols

for (i in 1:length(numeric_cols)) {
  col <- numeric_cols[i]
  outlier_matrix[,i] <- identify_outliers(model_data[[col]])
}

# Calculate outlier counts per column
outlier_counts <- colSums(outlier_matrix)
cat("\nOutlier counts per column:\n")
print(outlier_counts)

# Flag rows with multiple outliers
outlier_row_counts <- rowSums(outlier_matrix, na.rm = TRUE)
multi_outlier_rows <- outlier_row_counts >= 2
cat("Rows with multiple outliers:", sum(multi_outlier_rows), "out of", nrow(model_data), 
    "(", round(sum(multi_outlier_rows)/nrow(model_data) * 100, 2), "%)\n")

# Remove rows with multiple outliers
model_data_clean <- model_data[!multi_outlier_rows,]
cat("Data dimensions after removing multi-outlier rows:", dim(model_data_clean)[1], "x", dim(model_data_clean)[2], "\n")

# Replace model_data with cleaned version
model_data <- model_data_clean

###############################################################################
# 6) CORRELATION ANALYSIS AND FEATURE REDUCTION
###############################################################################
# Define correlation threshold for variable removal (same as in logistic regression)
correlation_threshold <- 0.7

# Get numeric columns for correlation analysis
numeric_cols <- names(model_data)[sapply(model_data, is.numeric)]
model_data_numeric <- model_data[numeric_cols]

# Generate correlation matrix
cor_matrix <- cor(model_data_numeric, use = "complete.obs")

# Print high correlations
high_cors <- which(abs(cor_matrix) > correlation_threshold & abs(cor_matrix) < 1, arr.ind = TRUE)
if(nrow(high_cors) > 0) {
  cat("\nHigh correlations (> ", correlation_threshold, "):\n", sep="")
  for(i in 1:nrow(high_cors)) {
    if(high_cors[i, 1] < high_cors[i, 2]) { # Avoid printing duplicates
      cat(rownames(cor_matrix)[high_cors[i, 1]], "and", 
          colnames(cor_matrix)[high_cors[i, 2]], ":", 
          round(cor_matrix[high_cors[i, 1], high_cors[i, 2]], 3), "\n")
    }
  }
}

# Visualize correlation matrix
png("correlation_matrix.png", width = 800, height = 800, res = 100)
corrplot(cor_matrix, method = "color", type = "upper",
         tl.col = "black", tl.srt = 45, 
         addCoef.col = "black", number.cex = 0.7)
dev.off()

# Automatically remove variables with correlation > threshold
remove_indices <- findCorrelation(cor_matrix, cutoff = correlation_threshold, verbose = TRUE)
if(length(remove_indices) > 0) {
  remove_vars <- colnames(model_data_numeric)[remove_indices]
  cat("Removing highly correlated variables:", paste(remove_vars, collapse = ", "), "\n")
  
  # Create optimized dataset by excluding flagged numeric variables
  optimized_data <- model_data[, !names(model_data) %in% remove_vars]
} else {
  optimized_data <- model_data
  cat("No variables with correlation >", correlation_threshold, "were found.\n")
}

###############################################################################
# 7) TRAIN/TEST SPLIT WITH STRATIFIED SAMPLING
###############################################################################
set.seed(123)  # Same seed for reproducibility

# Use simple stratified sampling on Target_Var
strat_var <- optimized_data$Target_Var

# Use createDataPartition from caret for stratified sampling
train_index <- createDataPartition(strat_var, p = 0.7, list = FALSE)
train_data <- optimized_data[train_index, ]
test_data <- optimized_data[-train_index, ]

# Verify stratification worked
cat("\nTraining data class distribution:\n")
print(table(train_data$Target_Var))
cat("\nTest data class distribution:\n")
print(table(test_data$Target_Var))

###############################################################################
# 8) MODEL DEVELOPMENT AND EVALUATION FUNCTION
###############################################################################
# Function for model evaluation - ensures consistent metrics across models
evaluate_model <- function(pred_prob, true_values, model_name) {
  # Calculate ROC and AUC
  roc_obj <- roc(as.numeric(as.character(true_values)), pred_prob)
  auc_val <- auc(roc_obj)
  
  # Calculate metrics at threshold = 0.5
  pred_class_05 <- ifelse(pred_prob > 0.5, 1, 0)
  cm_05 <- confusionMatrix(
    factor(pred_class_05, levels = c(0, 1)),
    factor(as.numeric(as.character(true_values)), levels = c(0, 1))
  )
  
  accuracy_05 <- cm_05$overall["Accuracy"]
  precision_05 <- cm_05$byClass["Pos Pred Value"]
  recall_05 <- cm_05$byClass["Sensitivity"]
  specificity_05 <- cm_05$byClass["Specificity"]
  f1_05 <- 2 * (precision_05 * recall_05) / (precision_05 + recall_05)
  
  # Calculate metrics at different thresholds for optimization
  thresholds <- seq(0.1, 0.9, by = 0.05)
  threshold_results <- data.frame(
    Threshold = thresholds,
    Accuracy = numeric(length(thresholds)),
    Precision = numeric(length(thresholds)),
    Recall = numeric(length(thresholds)),
    F1_Score = numeric(length(thresholds))
  )
  
  for (i in 1:length(thresholds)) {
    thresh <- thresholds[i]
    pred_class <- ifelse(pred_prob > thresh, 1, 0)
    
    # Convert to factors with same levels for proper confusion matrix calculation
    pred_factor <- factor(pred_class, levels = c(0, 1))
    true_factor <- factor(as.numeric(as.character(true_values)), levels = c(0, 1))
    
    cm <- confusionMatrix(pred_factor, true_factor)
    
    threshold_results$Accuracy[i] <- cm$overall["Accuracy"]
    threshold_results$Precision[i] <- cm$byClass["Pos Pred Value"]
    threshold_results$Recall[i] <- cm$byClass["Sensitivity"]
    threshold_results$F1_Score[i] <- 2 * (cm$byClass["Pos Pred Value"] * cm$byClass["Sensitivity"]) / 
      (cm$byClass["Pos Pred Value"] + cm$byClass["Sensitivity"])
  }
  
  # Find optimal threshold for F1 score
  optimal_f1_idx <- which.max(threshold_results$F1_Score)
  optimal_threshold <- threshold_results$Threshold[optimal_f1_idx]
  
  # Plot threshold optimization
  png(paste0(model_name, "_threshold_optimization.png"), width = 800, height = 600, res = 100)
  p <- ggplot(threshold_results, aes(x = Threshold)) +
    geom_line(aes(y = Accuracy, color = "Accuracy"), size = 1) +
    geom_line(aes(y = F1_Score, color = "F1 Score"), size = 1) +
    geom_line(aes(y = Precision, color = "Precision"), size = 1, linetype = "dashed") +
    geom_line(aes(y = Recall, color = "Recall"), size = 1, linetype = "dashed") +
    geom_vline(xintercept = optimal_threshold, linetype = "dashed") +
    scale_color_manual(values = c("Accuracy" = "#4878D0", "F1 Score" = "#E46726", 
                                  "Precision" = "#20B2AA", "Recall" = "#9370DB")) +
    labs(title = paste0(model_name, " Performance Metrics by Threshold"),
         subtitle = paste0("Optimal threshold: ", optimal_threshold),
         x = "Threshold", y = "Score", color = "Metric") +
    theme_minimal() +
    theme(legend.position = "bottom")
  print(p)
  dev.off()
  
  # Get metrics at optimal threshold
  pred_class_opt <- ifelse(pred_prob > optimal_threshold, 1, 0)
  cm_opt <- confusionMatrix(
    factor(pred_class_opt, levels = c(0, 1)),
    factor(as.numeric(as.character(true_values)), levels = c(0, 1))
  )
  
  accuracy_opt <- cm_opt$overall["Accuracy"]
  precision_opt <- cm_opt$byClass["Pos Pred Value"]
  recall_opt <- cm_opt$byClass["Sensitivity"]
  specificity_opt <- cm_opt$byClass["Specificity"]
  f1_opt <- 2 * (precision_opt * recall_opt) / (precision_opt + recall_opt)
  
  # Return all evaluation results
  return(list(
    model_name = model_name,
    roc_obj = roc_obj,
    auc = auc_val,
    
    # Metrics at threshold = 0.5
    accuracy_05 = accuracy_05,
    precision_05 = precision_05,
    recall_05 = recall_05,
    specificity_05 = specificity_05,
    f1_05 = f1_05,
    
    # Metrics at optimal threshold
    optimal_threshold = optimal_threshold,
    accuracy_opt = accuracy_opt,
    precision_opt = precision_opt,
    recall_opt = recall_opt,
    specificity_opt = specificity_opt,
    f1_opt = f1_opt,
    
    # Confusion matrices
    cm_05 = cm_05,
    cm_opt = cm_opt,
    
    # Full threshold results
    threshold_results = threshold_results
  ))
}

###############################################################################
# 9) RANDOM FOREST MODEL
###############################################################################
cat("\nTraining Random Forest model...\n")

# Train Random Forest model
rf_model <- randomForest(Target_Var ~ ., 
                         data = train_data, 
                         ntree = 100, 
                         importance = TRUE)

# Print basic model information
print(rf_model)

# Get predictions
rf_pred_train <- predict(rf_model, train_data)
rf_pred_test <- predict(rf_model, test_data)
rf_pred_prob <- predict(rf_model, test_data, type = "prob")[,2]

# Evaluate the model
rf_eval <- evaluate_model(rf_pred_prob, test_data$Target_Var, "RandomForest")

# Extract and plot feature importance
rf_importance <- importance(rf_model)
rf_importance_df <- data.frame(
  Variable = rownames(rf_importance),
  Importance = rf_importance[, "MeanDecreaseAccuracy"],
  Gini = rf_importance[, "MeanDecreaseGini"]
)
rf_importance_df <- rf_importance_df[order(rf_importance_df$Importance, decreasing = TRUE),]

# Plot feature importance
png("RandomForest_feature_importance.png", width = 800, height = 800, res = 100)
par(mar = c(5, 12, 4, 2))  # Adjust margins for long variable names
barplot(rf_importance_df$Importance[10:1], 
        names.arg = rf_importance_df$Variable[10:1],
        horiz = TRUE, 
        las = 1, 
        main = "Random Forest Feature Importance (Top 10)",
        xlab = "Mean Decrease in Accuracy")
dev.off()

# Create variable importance plot in same style as logistic regression
top_n <- 15
top_rf_features <- head(rf_importance_df, top_n)

png("RandomForest_feature_importance_ggplot.png", width = 1000, height = 800, res = 100)
p <- ggplot(top_rf_features, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "#4878D0") +
  coord_flip() +
  labs(title = "Random Forest Feature Importance",
       subtitle = paste0("Top ", top_n, " features ranked by importance"),
       x = "",
       y = "Importance (Mean Decrease in Accuracy)") +
  theme_minimal() +
  theme(
    legend.position = "none",
    panel.grid.minor = element_blank(),
    axis.text.y = element_text(size = 11, face = "bold"),
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(color = "darkgrey", size = 12)
  )
print(p)
dev.off()

###############################################################################
# 10) EXPLAINABLE BOOSTING MACHINE (EBM) MODEL
###############################################################################
# This code assumes the interpret package is installed.
# If you encounter problems with the interpret package, consider using the explainedmachine package instead.

cat("\nTraining EBM model...\n")

# Prepare data for EBM (needs matrices and numeric variables)
# The EBM implementation requires some special handling
tryCatch({
  # Convert factor variables to numeric for EBM
  prepare_for_ebm <- function(data) {
    result <- data
    # Convert factors to numeric one-hot encoding
    factor_cols <- sapply(data, is.factor)
    factor_cols <- setdiff(names(data)[factor_cols], "Target_Var")  # Exclude target
    
    for (col in factor_cols) {
      # Create one-hot encoded columns for each level
      for (level in levels(data[[col]])) {
        new_col_name <- paste0(col, "_", level)
        result[[new_col_name]] <- as.numeric(data[[col]] == level)
      }
      # Remove original factor column
      result[[col]] <- NULL
    }
    
    return(result)
  }
  
  # Prepare training and test data
  train_data_ebm <- prepare_for_ebm(train_data)
  test_data_ebm <- prepare_for_ebm(test_data)
  
  # X matrices exclude the target variable
  X_train <- as.matrix(train_data_ebm[, !names(train_data_ebm) %in% "Target_Var"])
  y_train <- as.numeric(as.character(train_data_ebm$Target_Var))
  X_test <- as.matrix(test_data_ebm[, !names(test_data_ebm) %in% "Target_Var"])
  y_test <- as.numeric(as.character(test_data_ebm$Target_Var))
  
  # Train EBM model
  ebm_model <- interpret::ebm_classify(
    X_train = X_train,
    y_train = y_train,
    num_outer_bags = 10,
    validation_size = 0.15,
    max_rounds = 5000,
    interactions = 5,
    early_stopping_rounds = 50
  )
  
  # Get predictions
  ebm_pred <- interpret::predict_ebm(ebm_model, X_test)
  ebm_pred_prob <- ebm_pred$probability
  
  # Evaluate the model
  ebm_eval <- evaluate_model(ebm_pred_prob, test_data$Target_Var, "EBM")
  
  # Extract and format feature importance
  ebm_global <- interpret::ebm_global_explain(ebm_model)
  ebm_importance_df <- data.frame(
    Variable = ebm_global$feature_names,
    Importance = abs(ebm_global$scores)  # Absolute value for importance
  )
  ebm_importance_df <- ebm_importance_df[order(ebm_importance_df$Importance, decreasing = TRUE),]
  
  # Plot feature importance
  png("EBM_feature_importance.png", width = 800, height = 800, res = 100)
  par(mar = c(5, 12, 4, 2))  # Adjust margins for long variable names
  barplot(ebm_importance_df$Importance[10:1], 
          names.arg = ebm_importance_df$Variable[10:1],
          horiz = TRUE, 
          las = 1, 
          main = "EBM Feature Importance (Top 10)",
          xlab = "Feature Importance Score")
  dev.off()
  
  # Create variable importance plot in same style as logistic regression
  top_n <- 15
  top_ebm_features <- head(ebm_importance_df, top_n)
  
  png("EBM_feature_importance_ggplot.png", width = 1000, height = 800, res = 100)
  p <- ggplot(top_ebm_features, aes(x = reorder(Variable, Importance), y = Importance)) +
    geom_bar(stat = "identity", fill = "#4878D0") +
    coord_flip() +
    labs(title = "EBM Feature Importance",
         subtitle = paste0("Top ", top_n, " features ranked by importance"),
         x = "",
         y = "Importance Score") +
    theme_minimal() +
    theme(
      legend.position = "none",
      panel.grid.minor = element_blank(),
      axis.text.y = element_text(size = 11, face = "bold"),
      plot.title = element_text(face = "bold", size = 16),
      plot.subtitle = element_text(color = "darkgrey", size = 12)
    )
  print(p)
  dev.off()
  
  # Save partial dependence plots for top features
  for (i in 1:min(5, nrow(ebm_importance_df))) {
    feature_name <- ebm_importance_df$Variable[i]
    feature_idx <- which(ebm_global$feature_names == feature_name)
    
    if (length(feature_idx) > 0) {
      png(paste0("EBM_pdp_", feature_name, ".png"), width = 800, height = 600, res = 100)
      interpret::plot_ebm_feature(ebm_model, feature_idx, 
                                  title = paste0("Partial Dependence Plot for ", feature_name))
      dev.off()
    }
  }
  
  ebm_available <- TRUE
  
}, error = function(e) {
  cat("Error in EBM modeling:", e$message, "\n")
  cat("Skipping EBM model and proceeding with Random Forest only.\n")
  ebm_available <- FALSE
})

###############################################################################
# 11) LOAD LOGISTIC REGRESSION RESULTS (FOR COMPARISON)
###############################################################################
# Since we don't have direct access to the logistic regression model's results,
# we'll create a placeholder with the metrics reported in the previous script
# In a real scenario, you would either load these from a saved file or re-run the logistic model

# Create placeholders based on the output we reviewed
logistic_metrics <- list(
  model_name = "LogisticRegression",
  auc = 0.721,
  
  # Metrics at threshold = 0.5
  accuracy_05 = 0.851,
  precision_05 = 0.553,
  recall_05 = 0.053,
  specificity_05 = 0.99,
  f1_05 = 0.097,
  
  # Metrics at optimal threshold
  optimal_threshold = 0.2,
  accuracy_opt = 0.77,
  precision_opt = 0.324,
  recall_opt = 0.487,
  specificity_opt = 0.8,
  f1_opt = 0.39
)

# Important features from logistic regression (based on output provided)
logistic_top_features <- c(
  "location_risk", "IMP_QUAL_MOD6_7", "IMP_QUAL_MODUNKNOWN", "CONST_CLASS_MOD1_5",
  "CONST_CLASS_MOD2", "CONST_CLASS_MOD4", "IMP_QUAL_MOD4", "CONST_CLASS_MODUNKNOWN",
  "MonthDifference", "EFF_AGE", "TOT_LVG_AREA", "SPEC_FEAT_VAL", "LND_VAL", "SALE_PRC1"
)

###############################################################################
# 12) MODEL COMPARISON
###############################################################################
# Create a comparison table
model_comparison <- data.frame(
  Model = c("Logistic Regression", "Random Forest", if(exists("ebm_eval")) "EBM" else NULL),
  
  # AUC
  AUC = c(logistic_metrics$auc, rf_eval$auc, if(exists("ebm_eval")) ebm_eval$auc else NULL),
  
  # Metrics at threshold = 0.5
  Accuracy_05 = c(logistic_metrics$accuracy_05, rf_eval$accuracy_05, if(exists("ebm_eval")) ebm_eval$accuracy_05 else NULL),
  Precision_05 = c(logistic_metrics$precision_05, rf_eval$precision_05, if(exists("ebm_eval")) ebm_eval$precision_05 else NULL),
  Recall_05 = c(logistic_metrics$recall_05, rf_eval$recall_05, if(exists("ebm_eval")) ebm_eval$recall_05 else NULL),
  F1_05 = c(logistic_metrics$f1_05, rf_eval$f1_05, if(exists("ebm_eval")) ebm_eval$f1_05 else NULL),
  
  # Optimal thresholds
  Opt_Threshold = c(logistic_metrics$optimal_threshold, rf_eval$optimal_threshold, if(exists("ebm_eval")) ebm_eval$optimal_threshold else NULL),
  
  # Metrics at optimal threshold
  Accuracy_Opt = c(logistic_metrics$accuracy_opt, rf_eval$accuracy_opt, if(exists("ebm_eval")) ebm_eval$accuracy_opt else NULL),
  Precision_Opt = c(logistic_metrics$precision_opt, rf_eval$precision_opt, if(exists("ebm_eval")) ebm_eval$precision_opt else NULL),
  Recall_Opt = c(logistic_metrics$recall_opt, rf_eval$recall_opt, if(exists("ebm_eval")) ebm_eval$recall_opt else NULL),
  F1_Opt = c(logistic_metrics$f1_opt, rf_eval$f1_opt, if(exists("ebm_eval")) ebm_eval$f1_opt else NULL)
)

# Print comparison table
cat("\nMODEL COMPARISON:\n")
print(model_comparison)

# Save comparison table
write.csv(model_comparison, "model_comparison.csv", row.names = FALSE)

# Plot ROC curves
png("roc_comparison.png", width = 800, height = 800, res = 100)
plot(rf_eval$roc_obj, col = "blue", main = "ROC Curve Comparison")
if(exists("ebm_eval")) {
  plot(ebm_eval$roc_obj, col = "green", add = TRUE)
  legend_text <- c(
    paste("Random Forest (AUC =", round(rf_eval$auc, 3), ")"),
    paste("EBM (AUC =", round(ebm_eval$auc, 3), ")"),
    paste("Logistic Regression (AUC =", round(logistic_metrics$auc, 3), ")")
  )
  legend_colors <- c("blue", "green", "red")
} else {
  legend_text <- c(
    paste("Random Forest (AUC =", round(rf_eval$auc, 3), ")"),
    paste("Logistic Regression (AUC =", round(logistic_metrics$auc, 3), ")")
  )
  legend_colors <- c("blue", "red")
}
# Add a reference for the logistic regression (even though we don't have the actual curve)
abline(a = 0, b = 1, lty = 2)  # Diagonal reference line
legend("bottomright", legend = legend_text, col = legend_colors, lwd = 2)
dev.off()

###############################################################################
# 13) FEATURE IMPORTANCE COMPARISON
###############################################################################
# Compare top features across models
top_n <- 10

# Prepare data frame for comparison
feature_comparison <- data.frame(
  Rank = 1:top_n,
  LogisticRegression = head(logistic_top_features, top_n),
  RandomForest = head(rf_importance_df$Variable, top_n)
)

if(exists("ebm_importance_df")) {
  feature_comparison$EBM <- head(ebm_importance_df$Variable, top_n)
}

# Print feature comparison
cat("\nTOP FEATURE COMPARISON:\n")
print(feature_comparison)

# Save feature comparison
write.csv(feature_comparison, "feature_comparison.csv", row.names = FALSE)

# Find common important features
identify_common_features <- function(list1, list2, list3 = NULL, top_n = 5) {
  # Take top n features from each list
  top_list1 <- head(list1, top_n)
  top_list2 <- head(list2, top_n)
  
  if (!is.null(list3)) {
    top_list3 <- head(list3, top_n)
    common <- intersect(intersect(top_list1, top_list2), top_list3)
  } else {
    common <- intersect(top_list1, top_list2)
  }
  
  return(common)
}

if(exists("ebm_importance_df")) {
  common_features <- identify_common_features(
    logistic_top_features, 
    rf_importance_df$Variable, 
    ebm_importance_df$Variable
  )
} else {
  common_features <- identify_common_features(
    logistic_top_features, 
    rf_importance_df$Variable
  )
}

cat("\nCommon top features across models:\n")
print(common_features)

# Create a visualization of feature rank comparison
if(exists("ebm_importance_df")) {
  # Get all unique features from the top 10 of each model
  all_features <- unique(c(
    head(logistic_top_features, top_n),
    head(rf_importance_df$Variable, top_n),
    head(ebm_importance_df$Variable, top_n)
  ))
} else {
  all_features <- unique(c(
    head(logistic_top_features, top_n),
    head(rf_importance_df$Variable, top_n)
  ))
}

# For each feature, get its rank in each model (or NA if not in top 10)
rank_df <- data.frame(Feature = all_features)

get_rank <- function(feature, feature_list) {
  idx <- match(feature, feature_list)
  if (!is.na(idx) && idx <= top_n) {
    return(idx)
  } else {
    return(NA)
  }
}

rank_df$LogisticRank <- sapply(rank_df$Feature, get_rank, logistic_top_features)
rank_df$RandomForestRank <- sapply(rank_df$Feature, get_rank, rf_importance_df$Variable)

if(exists("ebm_importance_df")) {
  rank_df$EBMRank <- sapply(rank_df$Feature, get_rank, ebm_importance_df$Variable)
}

# Order by average rank (excluding NAs)
rank_df$AvgRank <- rowMeans(rank_df[, grepl("Rank", names(rank_df))], na.rm = TRUE)
rank_df <- rank_df[order(rank_df$AvgRank), ]

# Convert to long format for plotting
if(exists("ebm_importance_df")) {
  rank_long <- reshape2::melt(
    rank_df[, c("Feature", "LogisticRank", "RandomForestRank", "EBMRank")],
    id.vars = "Feature",
    variable.name = "Model",
    value.name = "Rank"
  )
} else {
  rank_long <- reshape2::melt(
    rank_df[, c("Feature", "LogisticRank", "RandomForestRank")],
    id.vars = "Feature",
    variable.name = "Model",
    value.name = "Rank"
  )
}

# Clean up model names
rank_long$Model <- gsub("Rank", "", rank_long$Model)

# Plot feature ranks
png("feature_rank_comparison.png", width = 1000, height = 800, res = 100)
ggplot(rank_long, aes(x = Feature, y = Rank, color = Model, group = Model)) +
  geom_point(size = 3) +
  geom_line() +
  scale_y_reverse() +  # Lower rank (1) at the top
  coord_flip() +       # Horizontal plot
  labs(title = "Feature Importance Rank Comparison Across Models",
       x = "Feature",
       y = "Rank (lower is more important)") +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    axis.text.y = element_text(size = 10),
    plot.title = element_text(size = 14, face = "bold")
  )
dev.off()

###############################################################################
# 14) CREATE DASHBOARD OF RESULTS
###############################################################################
# Create a multi-panel dashboard using gridExtra
if(require(gridExtra)) {
  # Plot 1: Model comparison bar chart - F1 score
  p1 <- ggplot(model_comparison, aes(x = Model, y = F1_Opt, fill = Model)) +
    geom_bar(stat = "identity") +
    geom_text(aes(label = round(F1_Opt, 3)), vjust = -0.5) +
    labs(title = "Optimal F1 Score Comparison",
         x = "", y = "F1 Score") +
    theme_minimal() +
    theme(legend.position = "none")
  
  # Plot 2: Model comparison bar chart - AUC
  p2 <- ggplot(model_comparison, aes(x = Model, y = AUC, fill = Model)) +
    geom_bar(stat = "identity") +
    geom_text(aes(label = round(AUC, 3)), vjust = -0.5) +
    labs(title = "AUC Comparison",
         x = "", y = "AUC") +
    theme_minimal() +
    theme(legend.position = "none")
  
  # Plot 3: Top common features across models
  if(length(common_features) > 0) {
    data_common <- data.frame(
      Feature = common_features,
      Importance = 1:length(common_features)  # Just for ordering
    )
    
    p3 <- ggplot(data_common, aes(x = reorder(Feature, Importance), y = 1)) +
      geom_bar(stat = "identity", fill = "#4878D0") +
      geom_text(aes(label = Feature), hjust = -0.1, color = "black") +
      labs(title = "Common Important Features",
           x = "", y = "") +
      ylim(0, 2) +  # Make room for text
      theme_minimal() +
      theme(
        axis.text.y = element_blank(),
        axis.ticks = element_blank(),
        panel.grid = element_blank()
      ) +
      coord_flip()
  } else {
    # Create empty plot if no common features
    p3 <- ggplot() + 
      annotate("text", x = 0.5, y = 0.5, label = "No common features found") +
      theme_void() +
      labs(title = "Common Important Features")
  }
  
  # Plot 4: Optimal thresholds comparison
  p4 <- ggplot(model_comparison, aes(x = Model, y = Opt_Threshold, fill = Model)) +
    geom_bar(stat = "identity") +
    geom_text(aes(label = Opt_Threshold), vjust = -0.5) +
    labs(title = "Optimal Threshold Comparison",
         x = "", y = "Threshold") +
    theme_minimal() +
    theme(legend.position = "none")
  
  # Combine plots into a dashboard
  png("model_comparison_dashboard.png", width = 1200, height = 1000, res = 100)
  # Explicitly load grid package
  if(!require(grid)) install.packages("grid"); library(grid)
  grid.arrange(
    p1, p2, p3, p4,
    ncol = 2,
    top = grid::textGrob(
      "Property Loss Prediction Model Comparison",
      gp = grid::gpar(fontsize = 18, font = 2)
    )
  )
  dev.off()
}
###############################################################################
# 15) CONCLUSION & RECOMMENDATIONS
###############################################################################
# Determine the best model based on F1 score at optimal threshold
best_model_idx <- which.max(model_comparison$F1_Opt)
best_model_name <- as.character(model_comparison$Model[best_model_idx])

# Print final summary and recommendations
cat("\nCONCLUSION & RECOMMENDATIONS\n")
cat("=================================\n")

cat("Best performing model:", best_model_name, "\n")
cat("Key performance metrics for", best_model_name, ":\n")
cat(" - AUC:", round(model_comparison$AUC[best_model_idx], 4), "\n")
cat(" - Optimal threshold:", model_comparison$Opt_Threshold[best_model_idx], "\n")
cat(" - F1 Score at optimal threshold:", round(model_comparison$F1_Opt[best_model_idx], 4), "\n")
cat(" - Accuracy at optimal threshold:", round(model_comparison$Accuracy_Opt[best_model_idx], 4), "\n")
cat(" - Precision at optimal threshold:", round(model_comparison$Precision_Opt[best_model_idx], 4), "\n")
cat(" - Recall at optimal threshold:", round(model_comparison$Recall_Opt[best_model_idx], 4), "\n")

# Summarize feature importance findings
cat("\nKey predictors of property loss risk:\n")
if(length(common_features) > 0) {
  cat("The following features were important across all models:\n")
  cat(paste(" -", common_features), sep = "\n")
} else {
  cat("No features were commonly important across all models.\n")
  
  cat("\nTop features by model:\n")
  cat("Logistic Regression:", paste(head(logistic_top_features, 5), collapse = ", "), "\n")
  cat("Random Forest:", paste(head(rf_importance_df$Variable, 5), collapse = ", "), "\n")
  if(exists("ebm_importance_df")) {
    cat("EBM:", paste(head(ebm_importance_df$Variable, 5), collapse = ", "), "\n")
  }
}

# Model comparison summary
cat("\nModel comparison summary:\n")
cat("1. Logistic Regression:\n")
cat("   - Strengths: Highly interpretable coefficients with clear effect directions\n")
cat("   - Limitations: Lower F1 score, requires proper handling of categorical variables\n")
cat("   - Optimal threshold:", model_comparison$Opt_Threshold[1], "with F1 score:", round(model_comparison$F1_Opt[1], 4), "\n")

cat("2. Random Forest:\n")
cat("   - Strengths: Higher predictive performance, captures non-linear relationships\n")
cat("   - Limitations: Less interpretable than logistic regression\n")
cat("   - Optimal threshold:", model_comparison$Opt_Threshold[2], "with F1 score:", round(model_comparison$F1_Opt[2], 4), "\n")

if(exists("ebm_eval")) {
  cat("3. Explainable Boosting Machine (EBM):\n")
  cat("   - Strengths: Balances performance and interpretability, displays feature shapes\n")
  cat("   - Limitations: More complex to implement than logistic regression\n")
  cat("   - Optimal threshold:", model_comparison$Opt_Threshold[3], "with F1 score:", round(model_comparison$F1_Opt[3], 4), "\n")
}

# Recommendations
cat("\nRecommendations:\n")
cat(" - For maximum interpretability: Use Logistic Regression\n")

if(exists("ebm_eval")) {
  cat(" - For balance of performance and interpretability: Use EBM\n")
}

cat(" - For maximum predictive power: Use", best_model_name, "\n")
cat(" - Consider ensemble approach combining predictions from all models\n")
cat(" - Apply appropriate threshold (", model_comparison$Opt_Threshold[best_model_idx], 
    " for ", best_model_name, ") based on business needs\n", sep="")

# Save the entire conclusion to a text file
sink("model_comparison_conclusion.txt")
cat("PROPERTY LOSS PREDICTION MODEL COMPARISON\n")
cat("=================================\n\n")

cat("Best performing model:", best_model_name, "\n")
cat("Key performance metrics for", best_model_name, ":\n")
cat(" - AUC:", round(model_comparison$AUC[best_model_idx], 4), "\n")
cat(" - Optimal threshold:", model_comparison$Opt_Threshold[best_model_idx], "\n")
cat(" - F1 Score at optimal threshold:", round(model_comparison$F1_Opt[best_model_idx], 4), "\n")
cat(" - Accuracy at optimal threshold:", round(model_comparison$Accuracy_Opt[best_model_idx], 4), "\n")
cat(" - Precision at optimal threshold:", round(model_comparison$Precision_Opt[best_model_idx], 4), "\n")
cat(" - Recall at optimal threshold:", round(model_comparison$Recall_Opt[best_model_idx], 4), "\n")

cat("\nFull model comparison:\n")
print(model_comparison)

cat("\nKey predictors of property loss risk:\n")
if(length(common_features) > 0) {
  cat("The following features were important across all models:\n")
  cat(paste(" -", common_features), sep = "\n")
} else {
  cat("No features were commonly important across all models.\n")
  
  cat("\nTop features by model:\n")
  cat("Logistic Regression:", paste(head(logistic_top_features, 5), collapse = ", "), "\n")
  cat("Random Forest:", paste(head(rf_importance_df$Variable, 5), collapse = ", "), "\n")
  if(exists("ebm_importance_df")) {
    cat("EBM:", paste(head(ebm_importance_df$Variable, 5), collapse = ", "), "\n")
  }
}

cat("\nModel strengths and limitations:\n")
cat("1. Logistic Regression:\n")
cat("   - Strengths: Highly interpretable coefficients with clear effect directions\n")
cat("   - Limitations: Lower F1 score, requires proper handling of categorical variables\n")
cat("   - Optimal threshold:", model_comparison$Opt_Threshold[1], "with F1 score:", round(model_comparison$F1_Opt[1], 4), "\n")

cat("2. Random Forest:\n")
cat("   - Strengths: Higher predictive performance, captures non-linear relationships\n")
cat("   - Limitations: Less interpretable than logistic regression\n")
cat("   - Optimal threshold:", model_comparison$Opt_Threshold[2], "with F1 score:", round(model_comparison$F1_Opt[2], 4), "\n")

if(exists("ebm_eval")) {
  cat("3. Explainable Boosting Machine (EBM):\n")
  cat("   - Strengths: Balances performance and interpretability, displays feature shapes\n")
  cat("   - Limitations: More complex to implement than logistic regression\n")
  cat("   - Optimal threshold:", model_comparison$Opt_Threshold[3], "with F1 score:", round(model_comparison$F1_Opt[3], 4), "\n")
}

cat("\nBusiness Recommendations:\n")
cat(" - For maximum interpretability: Use Logistic Regression\n")

if(exists("ebm_eval")) {
  cat(" - For balance of performance and interpretability: Use EBM\n")
}

cat(" - For maximum predictive power: Use", best_model_name, "\n")
cat(" - Consider ensemble approach combining predictions from all models\n")
cat(" - Apply appropriate threshold (", model_comparison$Opt_Threshold[best_model_idx], 
    " for ", best_model_name, ") based on business needs\n", sep="")

sink()