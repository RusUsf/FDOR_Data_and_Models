################################################################################
# RANDOM FOREST MODEL FOR PROPERTY LOSS PREDICTION
# WITH CLASS IMBALANCE HANDLING, TUNING, AND DETAILED INTERPRETATION
###############################################################################
# This script demonstrates a full workflow:
# 1) Import and prepare data (including factor conversion)
# 2) Transform S_LEGAL into a numeric location risk score
# 3) Improve categorical variable handling based on actual counts
# 4) Perform outlier trimming based on IQR
# 5) Address class imbalance using sampling techniques
# 6) Split data into train/test with stratified sampling
# 7) Build and tune Random Forest model with hyperparameter optimization
# 8) Evaluate model performance with threshold optimization
# 9) Analyze feature importance and partial dependence
# 10) Compare with previous decision tree results

###############################################################################
# 1) LOAD REQUIRED PACKAGES
###############################################################################
# Load necessary libraries
if(!require(randomForest)) install.packages("randomForest"); library(randomForest)
if(!require(caret)) install.packages("caret"); library(caret)
if(!require(dplyr)) install.packages("dplyr"); library(dplyr)
if(!require(ggplot2)) install.packages("ggplot2"); library(ggplot2)
if(!require(pROC)) install.packages("pROC"); library(pROC)
if(!require(gridExtra)) install.packages("gridExtra"); library(gridExtra)
if(!require(pdp)) install.packages("pdp"); library(pdp)
if(!require(ROSE)) install.packages("ROSE"); library(ROSE) # For sampling methods
if(!require(doParallel)) install.packages("doParallel"); library(doParallel) # For parallel processing

# Set up parallel processing to speed up computations
num_cores <- max(1, parallel::detectCores() - 1)  # Leave one core free
cl <- makeCluster(num_cores)
registerDoParallel(cl)
cat("Parallel processing enabled with", num_cores, "cores\n")

###############################################################################
# 2) DATA IMPORT AND INITIAL PREPARATION
###############################################################################
# Set file path (adjust as needed)
raw_path <- r"(C:\Users\User\Desktop\Job Search\USF_School_Project\Integrative_Program_Project_Cohort_2022_MSBAIS\Model_Optimization_in_R\INPUTS.csv)"

# Import data and replace empty strings with 0
data_full <- read.csv(raw_path, stringsAsFactors = FALSE)
data_full[data_full == ""] <- 0

# Select only the columns needed for the analysis
data <- data_full %>% 
  select(LND_SQFOOT, TOT_LVG_AREA, S_LEGAL, CONST_CLASS, IMP_QUAL, JV, 
         LND_VAL, NO_BULDNG, NCONST_VAL, DEL_VAL, SPEC_FEAT_VAL, 
         MonthDifference, SALE_PRC1, Target_Var, EFF_AGE, ACT_AGE)

# Convert target variable to factor (binary classification)
data$Target_Var <- as.factor(data$Target_Var)

# Replace "0" in CONST_CLASS and IMP_QUAL with "NULL"
data$IMP_QUAL[data$IMP_QUAL == "0"] <- "NULL"
data$CONST_CLASS[data$CONST_CLASS == "0"] <- "NULL"

# Display the first few rows to verify the data
cat("Initial data dimensions:", dim(data)[1], "x", dim(data)[2], "\n")
cat("Class distribution in original data:\n")
print(table(data$Target_Var))
cat("Class imbalance ratio:", round(table(data$Target_Var)[1]/table(data$Target_Var)[2], 2), ":1\n")

# Create variable descriptions for better interpretability
var_descriptions <- c(
  "location_risk" = "Location Risk Score",
  "EFF_AGE" = "Effective Age of Property",
  "ACT_AGE" = "Actual Age of Property", 
  "LND_SQFOOT" = "Land Square Footage",
  "TOT_LVG_AREA" = "Total Living Area (sq ft)",
  "JV" = "Property Assessment Value",
  "LND_VAL" = "Land Value",
  "NO_BULDNG" = "Number of Buildings",
  "NCONST_VAL" = "New Construction Value",
  "DEL_VAL" = "Deletion Value",
  "SPEC_FEAT_VAL" = "Special Feature Value",
  "MonthDifference" = "Months Between Sales",
  "SALE_PRC1" = "Last Sale Price",
  "Target_Var" = "Property Loss Event"
)

# Create a mapping of IMP_QUAL levels to descriptive names
imp_qual_map <- c(
  "1" = "1 (Minimum/Low Cost)",
  "2" = "2 (Below Average)",
  "3" = "3 (Average)",
  "4" = "4 (Above Average)",
  "5" = "5 (Excellent)",
  "6" = "6 (Superior)",
  "7" = "7 (Not in FDOR standard)",
  "6_7" = "6-7 (Superior)",
  "UNKNOWN" = "Unknown/Missing Value"
)

# Create mapping of CONST_CLASS levels to descriptive names
const_class_map <- c(
  "1" = "1 (Fireproof Steel, Class A)",
  "2" = "2 (Reinforced Concrete, Class B)",
  "3" = "3 (Masonry, Class C)",
  "4" = "4 (Wood/Steel Studs, Class D)",
  "5" = "5 (Steel Frame w/Incomb. Walls, Class S)",
  "1_5" = "1&5 (Fireproof Steel & Steel Frame)",
  "UNKNOWN" = "Unknown/Missing Value"
)

# Add these mappings to our variable descriptions
for (level in names(imp_qual_map)) {
  var_descriptions[paste0("IMP_QUAL_MOD", level)] <- imp_qual_map[level]
}

for (level in names(const_class_map)) {
  var_descriptions[paste0("CONST_CLASS_MOD", level)] <- const_class_map[level]
}

###############################################################################
# 3) IMPROVED CATEGORICAL VARIABLE HANDLING BASED ON COUNTS
###############################################################################
cat("Improving categorical variable handling based on actual counts...\n")

# Print initial distributions
cat("\nOriginal IMP_QUAL distribution:\n")
table_imp_qual <- table(data$IMP_QUAL, useNA = "ifany")
print(table_imp_qual)

cat("\nOriginal CONST_CLASS distribution:\n")
table_const_class <- table(data$CONST_CLASS, useNA = "ifany")
print(table_const_class)

# IMP_QUAL handling improvements: Group rare categories and handle NULL values
data$IMP_QUAL_MOD <- data$IMP_QUAL
data$IMP_QUAL_MOD[data$IMP_QUAL_MOD %in% c("6", "7")] <- "6_7" # Combine rare values 6 and 7
data$IMP_QUAL_MOD[is.na(data$IMP_QUAL_MOD) | data$IMP_QUAL_MOD == "NULL"] <- "UNKNOWN" # Handle NULL/NA

# CONST_CLASS handling improvements: Group rare categories and handle NULL values
data$CONST_CLASS_MOD <- data$CONST_CLASS
data$CONST_CLASS_MOD[data$CONST_CLASS_MOD %in% c("1", "5")] <- "1_5" # Combine rare values 1 and 5
data$CONST_CLASS_MOD[is.na(data$CONST_CLASS_MOD) | data$CONST_CLASS_MOD == "NULL"] <- "UNKNOWN" # Handle NULL/NA

# Convert modified categorical variables to proper factors with explicit levels
data$IMP_QUAL_MOD <- factor(data$IMP_QUAL_MOD)
data$CONST_CLASS_MOD <- factor(data$CONST_CLASS_MOD)

# Check the new distributions
cat("\nModified IMP_QUAL_MOD distribution:\n")
table_imp_qual_mod <- table(data$IMP_QUAL_MOD, useNA = "ifany")
print(table_imp_qual_mod)

cat("\nModified CONST_CLASS_MOD distribution:\n")
table_const_class_mod <- table(data$CONST_CLASS_MOD, useNA = "ifany")
print(table_const_class_mod)

# Create bar plots of the original vs. modified categorical distributions
png("rf_categorical_improvements_imp_qual.png", width = 1000, height = 600, res = 100)
par(mfrow = c(1, 2))
barplot(table_imp_qual, main = "Original IMP_QUAL Distribution", 
        ylab = "Count", xlab = "IMP_QUAL", col = "skyblue", log = "y")
barplot(table_imp_qual_mod, main = "Modified IMP_QUAL Distribution", 
        ylab = "Count", xlab = "IMP_QUAL_MOD", col = "steelblue", log = "y")
dev.off()

png("rf_categorical_improvements_const_class.png", width = 1000, height = 600, res = 100)
par(mfrow = c(1, 2))
barplot(table_const_class, main = "Original CONST_CLASS Distribution", 
        ylab = "Count", xlab = "CONST_CLASS", col = "lightgreen", log = "y")
barplot(table_const_class_mod, main = "Modified CONST_CLASS Distribution", 
        ylab = "Count", xlab = "CONST_CLASS_MOD", col = "darkgreen", log = "y")
dev.off()

###############################################################################
# 4) TRANSFORM S_LEGAL INTO LOCATION RISK SCORE
###############################################################################
cat("Transforming S_LEGAL into location risk score...\n")

# Calculate loss rate by location
location_summary <- data %>%
  group_by(S_LEGAL) %>%
  summarize(
    location_risk = mean(as.numeric(as.character(Target_Var))),
    location_count = n()
  )

# Print summary of location data
cat("Total unique locations:", nrow(location_summary), "\n")
cat("Locations with at least 5 observations:", sum(location_summary$location_count >= 5), "\n")

# Only use locations with sufficient data (at least 5 observations)
# For rare locations, use the overall average risk
location_summary$location_risk[location_summary$location_count < 5] <- NA
avg_risk <- mean(as.numeric(as.character(data$Target_Var)))
cat("Average risk across all locations:", round(avg_risk * 100, 2), "%\n")

# Join back to the data
data <- left_join(data, location_summary[, c("S_LEGAL", "location_risk")], by = "S_LEGAL")
data$location_risk[is.na(data$location_risk)] <- avg_risk

# Create model data excluding original variables replaced with improved versions
model_data <- data[, !names(data) %in% c("S_LEGAL", "IMP_QUAL", "CONST_CLASS")]

# Create a histogram of location risk to understand its distribution
png("rf_location_risk_distribution.png", width = 800, height = 600, res = 100)
hist(data$location_risk, 
     main = "Distribution of Location Risk Scores",
     xlab = "Risk Score (probability of loss)",
     col = "steelblue",
     breaks = 30)
abline(v = avg_risk, col = "red", lwd = 2, lty = 2)
text(avg_risk + 0.02, max(hist(data$location_risk, plot = FALSE)$counts) * 0.8, 
     paste("Average:", round(avg_risk, 3)), col = "red")
dev.off()

# Analyze location risk distribution by target variable
cat("\nLocation risk statistics by target class:\n")
location_risk_stats <- aggregate(location_risk ~ Target_Var, data = model_data, 
                                 FUN = function(x) c(mean = mean(x), median = median(x), 
                                                     min = min(x), max = max(x)))
print(location_risk_stats)

# Create a boxplot of location risk by target variable
png("rf_location_risk_by_target.png", width = 800, height = 600, res = 100)
boxplot(location_risk ~ Target_Var, data = model_data,
        main = "Location Risk by Target Variable",
        xlab = "Target Variable (0 = No Loss, 1 = Loss)",
        ylab = "Location Risk Score",
        col = c("skyblue", "coral"))
dev.off()

###############################################################################
# 5) OUTLIER TRIMMING BASED ON IQR
###############################################################################
cat("Identifying and removing outliers based on IQR...\n")

# Function to identify outliers based on IQR
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
numeric_cols_for_outliers <- names(model_data)[sapply(model_data, is.numeric)]
numeric_cols_for_outliers <- setdiff(numeric_cols_for_outliers, c("Target_Var", "location_risk_bin")) # Exclude non-numeric columns

# Create outlier matrix
outlier_matrix <- matrix(FALSE, nrow = nrow(model_data), ncol = length(numeric_cols_for_outliers))
colnames(outlier_matrix) <- numeric_cols_for_outliers

# Identify outliers for each column
for (i in 1:length(numeric_cols_for_outliers)) {
  col <- numeric_cols_for_outliers[i]
  outlier_matrix[,i] <- identify_outliers(model_data[[col]])
}

# Calculate outlier counts per column
outlier_counts <- colSums(outlier_matrix, na.rm = TRUE)
cat("\nOutlier counts per variable:\n")
print(outlier_counts)

# Flag rows with multiple outliers
outlier_row_counts <- rowSums(outlier_matrix, na.rm = TRUE)
multi_outlier_rows <- outlier_row_counts >= 2

cat("Rows with multiple outliers:", sum(multi_outlier_rows), "out of", nrow(model_data), 
    "(", round(sum(multi_outlier_rows)/nrow(model_data) * 100, 2), "%)\n")

# Remove rows with multiple outliers
model_data_clean <- model_data[!multi_outlier_rows,]
cat("Data dimensions after removing multi-outlier rows:", dim(model_data_clean)[1], "x", dim(model_data_clean)[2], "\n")

# Check class distribution after outlier removal
cat("Class distribution after outlier removal:\n")
print(table(model_data_clean$Target_Var))
cat("Class imbalance ratio after outlier removal:", 
    round(table(model_data_clean$Target_Var)[1]/table(model_data_clean$Target_Var)[2], 2), ":1\n")

# Create a histogram to see outlier distribution by row
png("rf_outlier_distribution.png", width = 800, height = 600, res = 100)
hist(outlier_row_counts, 
     main = "Distribution of Outliers per Row",
     xlab = "Number of Variables with Outliers",
     col = "steelblue",
     breaks = max(outlier_row_counts))
abline(v = 2, col = "red", lwd = 2, lty = 2)
text(2 + 0.2, max(hist(outlier_row_counts, plot = FALSE)$counts) * 0.8, 
     "Removal Threshold", col = "red")
dev.off()

# Replace model_data with cleaned version
model_data <- model_data_clean
model_data$location_risk_bin <- NULL  # Remove temporary bin column before modeling

###############################################################################
# 6) ADDRESS CLASS IMBALANCE
###############################################################################
cat("\nAddressing class imbalance...\n")

# Function to calculate and print class metrics for a dataset
print_class_metrics <- function(data, label) {
  class_counts <- table(data$Target_Var)
  imbalance_ratio <- class_counts[1] / class_counts[2]
  cat(label, "class distribution:", class_counts[1], "vs", class_counts[2], 
      "(ratio:", round(imbalance_ratio, 2), ":1)\n")
}

# Original class distribution 
print_class_metrics(model_data, "Original")

# Store original data before sampling
original_data <- model_data

# Sample data using down-sampling, up-sampling, and SMOTE
set.seed(42)

# Apply down-sampling to balance the classes
down_data <- downSample(x = model_data[, -which(names(model_data) == "Target_Var")], 
                        y = model_data$Target_Var,
                        yname = "Target_Var")
print_class_metrics(down_data, "Down-sampled")

# Apply up-sampling to balance the classes
up_data <- upSample(x = model_data[, -which(names(model_data) == "Target_Var")], 
                    y = model_data$Target_Var,
                    yname = "Target_Var")
print_class_metrics(up_data, "Up-sampled")

# Apply SMOTE (Synthetic Minority Over-sampling Technique)
# This creates synthetic examples of the minority class
set.seed(42)
smote_data <- ROSE::ROSE(Target_Var ~ ., data = model_data, seed = 42)$data
print_class_metrics(smote_data, "SMOTE")

# Visualize class distribution before and after sampling
sampling_results <- data.frame(
  Sampling_Method = c("Original", "Down-sampling", "Up-sampling", "SMOTE"),
  Class_0 = c(table(model_data$Target_Var)[1], 
              table(down_data$Target_Var)[1], 
              table(up_data$Target_Var)[1], 
              table(smote_data$Target_Var)[1]),
  Class_1 = c(table(model_data$Target_Var)[2], 
              table(down_data$Target_Var)[2], 
              table(up_data$Target_Var)[2], 
              table(smote_data$Target_Var)[2])
)

sampling_results$Ratio <- sampling_results$Class_0 / sampling_results$Class_1
sampling_results$Total <- sampling_results$Class_0 + sampling_results$Class_1

print(sampling_results)

# Create a stacked bar chart of class distribution
sampling_plot_data <- data.frame(
  Method = rep(sampling_results$Sampling_Method, 2),
  Class = c(rep("No Loss (0)", nrow(sampling_results)), rep("Loss (1)", nrow(sampling_results))),
  Count = c(sampling_results$Class_0, sampling_results$Class_1)
)

png("rf_class_balancing_results.png", width = 1000, height = 600, res = 100)
ggplot(sampling_plot_data, aes(x = Method, y = Count, fill = Class)) +
  geom_bar(stat = "identity", position = "stack") +
  scale_fill_manual(values = c("No Loss (0)" = "steelblue", "Loss (1)" = "coral")) +
  labs(title = "Class Distribution by Sampling Method",
       x = "Sampling Method",
       y = "Count",
       fill = "Class") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(face = "bold")) +
  geom_text(aes(label = Count), position = position_stack(vjust = 0.5))
dev.off()

# We'll use SMOTE data as our primary training set, 
# but will also compare model performance with other approaches
balanced_data <- smote_data

# Store the optimized dataset (without correlated variables) for further processing
original_data <- optimized_data

###############################################################################
# 7) TRAIN/TEST SPLIT WITH STRATIFIED SAMPLING
###############################################################################
cat("\nSplitting data into training and test sets...\n")

# For test data, we'll use the original (imbalanced) data to evaluate real-world performance
# For training, we'll use the balanced data to help the model learn both classes

# First, split the original data
set.seed(123)
# Create stratification variable that combines Target_Var and key categorical variables
strat_var <- with(original_data, paste(Target_Var, IMP_QUAL_MOD, CONST_CLASS_MOD))

# Use createDataPartition from caret for stratified sampling
train_index <- createDataPartition(strat_var, p = 0.7, list = FALSE)
original_train <- original_data[train_index, ]
test_data <- original_data[-train_index, ]

# Now prepare our training sets from different sampling approaches
# For each balanced dataset, we'll only use records that correspond to the original train set
down_train <- downSample(x = original_train[, -which(names(original_train) == "Target_Var")], 
                         y = original_train$Target_Var,
                         yname = "Target_Var")

up_train <- upSample(x = original_train[, -which(names(original_train) == "Target_Var")], 
                     y = original_train$Target_Var,
                     yname = "Target_Var")

set.seed(42)
smote_train <- ROSE::ROSE(Target_Var ~ ., data = original_train, seed = 42)$data

# Verify class distributions in train/test sets
cat("\nClass distributions in training/test sets:\n")
cat("Original train set: ")
print(table(original_train$Target_Var))
cat("Downsampled train set: ")
print(table(down_train$Target_Var))
cat("Upsampled train set: ")
print(table(up_train$Target_Var))
cat("SMOTE train set: ")
print(table(smote_train$Target_Var))
cat("Test set: ")
print(table(test_data$Target_Var))

# We'll primarily use SMOTE for training, but will compare with other approaches
train_data <- smote_train

###############################################################################
# 8) RANDOM FOREST MODEL BUILDING WITH HYPERPARAMETER TUNING
###############################################################################
cat("\nTraining Random Forest models with different sampling approaches...\n")

# Function to train a Random Forest model and return it
train_rf_model <- function(train_data, mtry = NULL, ntree = 500, importance = TRUE) {
  if (is.null(mtry)) {
    mtry <- floor(sqrt(ncol(train_data) - 1))  # Default mtry for classification
  }
  
  rf_model <- randomForest(
    Target_Var ~ .,
    data = train_data,
    mtry = mtry,          # Number of variables randomly sampled at each split
    ntree = ntree,        # Number of trees in the forest
    importance = importance,  # Calculate variable importance
    sampsize = table(train_data$Target_Var), # Sample sizes for each class
    replace = TRUE,       # Sample with replacement
    do.trace = FALSE      # Don't show progress
  )
  
  return(rf_model)
}

# Check if correlated variables were actually removed
cat("\nVerifying correlated variables were removed before modeling...\n")
cat("ACT_AGE still in training data:", "ACT_AGE" %in% names(original_train), "\n")
cat("JV still in training data:", "JV" %in% names(original_train), "\n")



# Train models with different sampling approaches
set.seed(123)
cat("Training Random Forest with original (imbalanced) data...\n")
rf_original <- train_rf_model(original_train)
cat("Training Random Forest with down-sampled data...\n")
rf_down <- train_rf_model(down_train)
cat("Training Random Forest with up-sampled data...\n")
rf_up <- train_rf_model(up_train)
cat("Training Random Forest with SMOTE data...\n")
rf_smote <- train_rf_model(smote_train)

# Print model information
cat("\nRandom Forest Model Summary (SMOTE):\n")
print(rf_smote)

# Hyperparameter tuning for the SMOTE model
cat("\nPerforming hyperparameter tuning for SMOTE model...\n")

# Define parameter grid for tuning
param_grid <- expand.grid(
  mtry = c(2, 4, 6, 8),   # Number of variables at each split
  ntree = c(100, 300, 500) # Number of trees
)

# Train models with different parameter combinations and track performance
tune_results <- data.frame(
  mtry = numeric(),
  ntree = numeric(),
  accuracy = numeric(),
  kappa = numeric(),
  sensitivity = numeric(),
  specificity = numeric(),
  auc = numeric()
)

# Perform 5-fold cross-validation
set.seed(123)
folds <- createFolds(smote_train$Target_Var, k = 5, list = TRUE, returnTrain = FALSE)

for (i in 1:nrow(param_grid)) {
  mtry_val <- param_grid$mtry[i]
  ntree_val <- param_grid$ntree[i]
  
  cat("Testing mtry =", mtry_val, "and ntree =", ntree_val, "\n")
  
  # Store results for each fold
  fold_metrics <- data.frame(
    accuracy = numeric(length(folds)),
    kappa = numeric(length(folds)),
    sensitivity = numeric(length(folds)),
    specificity = numeric(length(folds)),
    auc = numeric(length(folds))
  )
  
  # Cross-validation
  for (j in 1:length(folds)) {
    # Split data into training and validation sets
    valid_indices <- folds[[j]]
    cv_train <- smote_train[-valid_indices, ]
    cv_valid <- smote_train[valid_indices, ]
    
    # Train model
    rf_cv <- train_rf_model(cv_train, mtry = mtry_val, ntree = ntree_val)
    
    # Make predictions
    preds <- predict(rf_cv, cv_valid, type = "class")
    probs <- predict(rf_cv, cv_valid, type = "prob")[, "1"]
    
    # Calculate performance metrics
    cm <- confusionMatrix(preds, cv_valid$Target_Var)
    roc_obj <- roc(as.numeric(as.character(cv_valid$Target_Var)), probs)
    
    # Store fold results
    fold_metrics$accuracy[j] <- cm$overall["Accuracy"]
    fold_metrics$kappa[j] <- cm$overall["Kappa"]
    fold_metrics$sensitivity[j] <- cm$byClass["Sensitivity"]
    fold_metrics$specificity[j] <- cm$byClass["Specificity"]
    fold_metrics$auc[j] <- auc(roc_obj)
  }
  
  # Calculate mean performance across folds
  tune_results <- rbind(tune_results, data.frame(
    mtry = mtry_val,
    ntree = ntree_val,
    accuracy = mean(fold_metrics$accuracy),
    kappa = mean(fold_metrics$kappa),
    sensitivity = mean(fold_metrics$sensitivity),
    specificity = mean(fold_metrics$specificity),
    auc = mean(fold_metrics$auc)
  ))
}

# Find best parameters based on AUC (best balance of sensitivity and specificity)
best_params <- tune_results[which.max(tune_results$auc), ]
cat("\nBest parameters based on AUC:\n")
print(best_params)

# Visualize parameter tuning results
png("rf_parameter_tuning.png", width = 1000, height = 800, res = 100)
ggplot(tune_results, aes(x = as.factor(mtry), y = as.factor(ntree), fill = auc)) +
  geom_tile() +
  geom_text(aes(label = round(auc, 3)), color = "white", size = 3) +
  scale_fill_gradient(low = "steelblue", high = "red") +
  labs(title = "Random Forest Hyperparameter Tuning",
       subtitle = "Area Under ROC Curve (AUC) for different parameter combinations",
       x = "mtry (Variables per Split)",
       y = "ntree (Number of Trees)",
       fill = "AUC") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))
dev.off()

# Train final model with best parameters
cat("\nTraining final Random Forest model with best parameters...\n")
final_rf <- train_rf_model(smote_train, 
                           mtry = best_params$mtry, 
                           ntree = best_params$ntree,
                           importance = TRUE)

###############################################################################
# 9) MODEL EVALUATION WITH THRESHOLD OPTIMIZATION
###############################################################################
cat("\nEvaluating model performance on test data...\n")

# Function to evaluate model on test data and return metrics
evaluate_model <- function(model, test_data, model_name, threshold = 0.5) {
  # Get predictions
  pred_probs <- predict(model, test_data, type = "prob")[, "1"]
  pred_class <- ifelse(pred_probs > threshold, "1", "0")
  pred_class <- factor(pred_class, levels = levels(test_data$Target_Var))
  
  # Calculate metrics
  cm <- confusionMatrix(pred_class, test_data$Target_Var)
  roc_obj <- roc(as.numeric(as.character(test_data$Target_Var)), pred_probs)
  auc_val <- auc(roc_obj)
  
  # Create results dataframe
  results <- data.frame(
    Model = model_name,
    Accuracy = cm$overall["Accuracy"],
    Sensitivity = cm$byClass["Sensitivity"],
    Specificity = cm$byClass["Specificity"],
    BalancedAccuracy = cm$byClass["Balanced Accuracy"],
    AUC = auc_val,
    Threshold = threshold,
    stringsAsFactors = FALSE
  )
  
  # Return both metrics and ROC object for later use
  return(list(metrics = results, roc = roc_obj, cm = cm, probs = pred_probs))
}

# Evaluate models on test data with default threshold of 0.5
results_original <- evaluate_model(rf_original, test_data, "Original")
results_down <- evaluate_model(rf_down, test_data, "Down-sample")
results_up <- evaluate_model(rf_up, test_data, "Up-sample")
results_smote <- evaluate_model(rf_smote, test_data, "SMOTE")
results_final <- evaluate_model(final_rf, test_data, "Final-Tuned")

# Combine all results
all_results <- rbind(
  results_original$metrics,
  results_down$metrics,
  results_up$metrics,
  results_smote$metrics,
  results_final$metrics
)

# Print combined results
cat("\nModel Performance Comparison (threshold = 0.5):\n")
print(all_results)

# Visualize model performance comparison
png("rf_model_comparison.png", width = 1000, height = 600, res = 100)
comparison_long <- reshape2::melt(all_results, 
                                  id.vars = c("Model", "Threshold"),
                                  variable.name = "Metric", 
                                  value.name = "Value")

ggplot(comparison_long[comparison_long$Metric %in% c("Accuracy", "Sensitivity", "Specificity", "AUC"),], 
       aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_brewer(palette = "Set1") +
  labs(title = "Random Forest Model Performance Comparison",
       subtitle = "Different sampling approaches with default threshold (0.5)",
       x = "", y = "Score") +
  ylim(0, 1) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(face = "bold"))
dev.off()

# Plot ROC curves for all models
png("rf_roc_comparison.png", width = 800, height = 800, res = 100)
plot(results_original$roc, col = "black", lwd = 2, main = "ROC Curves for Different Sampling Methods")
plot(results_down$roc, col = "blue", lwd = 2, add = TRUE)
plot(results_up$roc, col = "red", lwd = 2, add = TRUE)
plot(results_smote$roc, col = "green", lwd = 2, add = TRUE)
plot(results_final$roc, col = "purple", lwd = 2, add = TRUE)
legend("bottomright", 
       legend = c(paste("Original:", round(results_original$metrics$AUC, 3)),
                  paste("Down-sample:", round(results_down$metrics$AUC, 3)),
                  paste("Up-sample:", round(results_up$metrics$AUC, 3)),
                  paste("SMOTE:", round(results_smote$metrics$AUC, 3)),
                  paste("Final-Tuned:", round(results_final$metrics$AUC, 3))),
       col = c("black", "blue", "red", "green", "purple"),
       lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "gray")
dev.off()

# Perform threshold optimization for the final model
cat("\nPerforming threshold optimization for the final model...\n")
thresholds <- seq(0.1, 0.9, by = 0.05)
threshold_results <- data.frame(
  Threshold = thresholds,
  Accuracy = numeric(length(thresholds)),
  Sensitivity = numeric(length(thresholds)),
  Specificity = numeric(length(thresholds)),
  BalancedAccuracy = numeric(length(thresholds)),
  F1Score = numeric(length(thresholds))
)

# Calculate metrics for each threshold
for (i in 1:length(thresholds)) {
  thresh <- thresholds[i]
  pred_class <- ifelse(results_final$probs > thresh, "1", "0")
  pred_class <- factor(pred_class, levels = levels(test_data$Target_Var))
  
  cm <- confusionMatrix(pred_class, test_data$Target_Var)
  
  threshold_results$Accuracy[i] <- cm$overall["Accuracy"]
  threshold_results$Sensitivity[i] <- cm$byClass["Sensitivity"]
  threshold_results$Specificity[i] <- cm$byClass["Specificity"]
  threshold_results$BalancedAccuracy[i] <- cm$byClass["Balanced Accuracy"]
  
  # Calculate F1 Score
  precision <- cm$byClass["Pos Pred Value"]
  recall <- cm$byClass["Sensitivity"]
  threshold_results$F1Score[i] <- 2 * (precision * recall) / (precision + recall)
  if (is.na(threshold_results$F1Score[i])) threshold_results$F1Score[i] <- 0
}

# Find optimal threshold based on balanced accuracy
optimal_idx <- which.max(threshold_results$BalancedAccuracy)
optimal_threshold <- threshold_results$Threshold[optimal_idx]

cat("\nOptimal threshold based on balanced accuracy:", optimal_threshold, "\n")
cat("Metrics at optimal threshold:\n")
print(threshold_results[optimal_idx, ])

# Visualize threshold optimization
png("rf_threshold_optimization.png", width = 1000, height = 600, res = 100)
threshold_long <- reshape2::melt(threshold_results, 
                                 id.vars = "Threshold",
                                 variable.name = "Metric",
                                 value.name = "Value")

ggplot(threshold_long, aes(x = Threshold, y = Value, color = Metric)) +
  geom_line(size = 1) +
  geom_point() +
  geom_vline(xintercept = optimal_threshold, linetype = "dashed", color = "black") +
  annotate("text", x = optimal_threshold + 0.05, y = 0.5, 
           label = paste("Optimal =", optimal_threshold), angle = 90) +
  labs(title = "Threshold Optimization for Random Forest Model",
       subtitle = "Finding the optimal balance between sensitivity and specificity",
       x = "Probability Threshold",
       y = "Metric Value") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))
dev.off()

# Re-evaluate the final model with optimal threshold
final_eval <- evaluate_model(final_rf, test_data, "Final-Optimal", threshold = optimal_threshold)

cat("\nFinal model performance with optimal threshold:\n")
print(final_eval$metrics)

# Create confusion matrix visualization
cm_df <- as.data.frame(final_eval$cm$table)
colnames(cm_df) <- c("Predicted", "Actual", "Freq")

png("rf_confusion_matrix.png", width = 800, height = 700, res = 100)
ggplot(cm_df, aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 15, fontface = "bold") +
  scale_fill_gradient(low = "#4477AA", high = "#AA4444") +
  labs(title = "Confusion Matrix - Final Random Forest Model",
       subtitle = paste("Threshold =", optimal_threshold),
       x = "Predicted Class",
       y = "Actual Class") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14),
        axis.title = element_text(size = 12),
        axis.text = element_text(size = 12, face = "bold"))
dev.off()

###############################################################################
# 10) FEATURE IMPORTANCE ANALYSIS
###############################################################################
cat("\nAnalyzing feature importance...\n")

# Extract variable importance from the final model
importance_matrix <- importance(final_rf)
importance_df <- data.frame(
  Variable = rownames(importance_matrix),
  MeanDecreaseAccuracy = importance_matrix[, "MeanDecreaseAccuracy"],
  MeanDecreaseGini = importance_matrix[, "MeanDecreaseGini"],
  stringsAsFactors = FALSE
)

# Sort by Mean Decrease in Accuracy (more interpretable measure)
importance_df <- importance_df[order(importance_df$MeanDecreaseAccuracy, decreasing = TRUE), ]

# Add descriptive names with more comprehensive handling
importance_df$Description <- sapply(importance_df$Variable, function(var_name) {
  # Check if it's in our basic variable descriptions
  if(var_name %in% names(var_descriptions)) {
    return(var_descriptions[var_name])
  } 
  # Check if it's a construction class variable
  else if(startsWith(var_name, "CONST_CLASS_MOD")) {
    class_level <- gsub("CONST_CLASS_MOD", "", var_name)
    if(class_level %in% names(const_class_map)) {
      return(paste("Construction Class:", const_class_map[class_level]))
    }
  }
  # Check if it's an improvement quality variable
  else if(startsWith(var_name, "IMP_QUAL_MOD")) {
    quality_level <- gsub("IMP_QUAL_MOD", "", var_name)
    if(quality_level %in% names(imp_qual_map)) {
      return(paste("Improvement Quality:", imp_qual_map[quality_level]))
    }
  }
  # Default fallback
  return(var_name)
})
# Print top features based on importance
cat("\nTop 10 most important variables in the Random Forest model:\n")
print(head(importance_df, 10))

# Visualize variable importance
png("rf_variable_importance.png", width = 1000, height = 800, res = 100)
ggplot(head(importance_df, 15), aes(x = reorder(Description, MeanDecreaseAccuracy), y = MeanDecreaseAccuracy)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Variable Importance in Random Forest Model",
       subtitle = "Based on Mean Decrease in Accuracy (higher = more important)",
       x = "",
       y = "Mean Decrease in Accuracy") +
  theme_minimal() +
  theme(
    axis.text.y = element_text(size = 10),
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(color = "darkgrey", size = 12)
  )
dev.off()

# Create a second importance plot based on Gini importance
png("rf_variable_importance_gini.png", width = 1000, height = 800, res = 100)
importance_df_gini <- importance_df[order(importance_df$MeanDecreaseGini, decreasing = TRUE), ]
ggplot(head(importance_df_gini, 15), aes(x = reorder(Description, MeanDecreaseGini), y = MeanDecreaseGini)) +
  geom_bar(stat = "identity", fill = "darkgreen") +
  coord_flip() +
  labs(title = "Variable Importance in Random Forest Model",
       subtitle = "Based on Mean Decrease in Gini (node purity)",
       x = "",
       y = "Mean Decrease in Gini") +
  theme_minimal() +
  theme(
    axis.text.y = element_text(size = 10),
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(color = "darkgrey", size = 12)
  )
dev.off()

###############################################################################
# 11) PARTIAL DEPENDENCE PLOTS FOR KEY VARIABLES
###############################################################################
cat("\nCreating partial dependence plots for key variables...\n")

# Select top 5 variables for partial dependence plots
top_vars <- head(importance_df, 5)$Variable
pdp_plots <- list()

for (var in top_vars) {
  # Skip factor variables as they require special handling
  if (!is.numeric(train_data[[var]])) {
    cat("Skipping non-numeric variable:", var, "\n")
    next
  }
  
  # Create partial dependence plot
  pdp_result <- partial(final_rf, pred.var = var, train = train_data,
                        prob = TRUE, plot = FALSE, plot.engine = "ggplot2")
  
  # Get nice variable name
  var_name <- if(var %in% names(var_descriptions)) var_descriptions[var] else var
  
  # Plot
  p <- ggplot(pdp_result, aes(x = pdp_result[[1]], y = yhat)) +
    geom_line(size = 1.2, color = "steelblue") +
    geom_rug(sides = "b", alpha = 0.3) +
    labs(title = paste("Partial Dependence Plot for", var_name),
         subtitle = "How the variable affects the predicted probability of property loss",
         x = var_name,
         y = "Predicted Probability of Loss") +
    theme_minimal() +
    theme(plot.title = element_text(face = "bold"))
  
  # Store plot
  pdp_plots[[var]] <- p
  
  # Save individual plot
  png(paste0("rf_pdp_", var, ".png"), width = 800, height = 500, res = 100)
  print(p)
  dev.off()
}

# Create a combined plot of top 4 numeric variables
numeric_top_vars <- importance_df$Variable[sapply(importance_df$Variable, function(v) is.numeric(train_data[[v]]))]
numeric_top_vars <- head(numeric_top_vars, 4)

if (length(numeric_top_vars) > 0) {
  png("rf_pdp_combined.png", width = 1200, height = 1000, res = 100)
  if (length(numeric_top_vars) >= 4) {
    grid.arrange(pdp_plots[[numeric_top_vars[1]]], 
                 pdp_plots[[numeric_top_vars[2]]], 
                 pdp_plots[[numeric_top_vars[3]]], 
                 pdp_plots[[numeric_top_vars[4]]], 
                 ncol = 2)
  } else if (length(numeric_top_vars) >= 2) {
    grid.arrange(pdp_plots[[numeric_top_vars[1]]], 
                 pdp_plots[[numeric_top_vars[2]]], 
                 ncol = 2)
  } else if (length(numeric_top_vars) == 1) {
    print(pdp_plots[[numeric_top_vars[1]]])
  }
  dev.off()
}

###############################################################################
# 12) COMPARISON WITH DECISION TREE RESULTS
###############################################################################
cat("\nComparing Random Forest with Decision Tree results...\n")

# Define decision tree performance metrics (from previous model)
# Replace these values with your actual decision tree results
dt_accuracy <- 0.85      # Observed from the previous model report
dt_sensitivity <- 0.0    # Observed from the previous model report
dt_specificity <- 1.0    # Observed from the previous model report
dt_auc <- 0.5            # Observed from the previous model report

# Create comparison dataframe
model_comparison <- data.frame(
  Metric = c("Accuracy", "Sensitivity", "Specificity", "Balanced Accuracy", "AUC"),
  DecisionTree = c(dt_accuracy, dt_sensitivity, dt_specificity, 
                   (dt_sensitivity + dt_specificity)/2, dt_auc),
  RandomForest = c(final_eval$metrics$Accuracy, 
                   final_eval$metrics$Sensitivity, 
                   final_eval$metrics$Specificity,
                   final_eval$metrics$BalancedAccuracy,
                   final_eval$metrics$AUC)
)

# Print comparison
cat("\nModel Comparison - Decision Tree vs. Random Forest:\n")
print(model_comparison)

# Visualize comparison
png("rf_vs_dt_comparison.png", width = 1000, height = 700, res = 100)
comparison_long <- reshape2::melt(model_comparison, 
                                  id.vars = "Metric", 
                                  variable.name = "Model", 
                                  value.name = "Value")

ggplot(comparison_long, aes(x = Metric, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.9)) +
  geom_text(aes(label = sprintf("%.3f", Value)), 
            position = position_dodge(width = 0.9),
            vjust = -0.5, size = 3.5) +
  labs(title = "Model Performance Comparison",
       subtitle = "Decision Tree vs. Random Forest with Class Imbalance Handling",
       y = "Score",
       x = "") +
  ylim(0, 1) +
  scale_fill_manual(values = c("DecisionTree" = "coral", "RandomForest" = "steelblue")) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    axis.text.x = element_text(angle = 0, hjust = 0.5, face = "bold"),
    plot.title = element_text(face = "bold", size = 14)
  )
dev.off()

###############################################################################
# 13) ADDITIONAL ANALYSIS: SAMPLING METHODS COMPARISON FOR IMBALANCED DATA
###############################################################################
cat("\n\nSAMPLING METHODS COMPARISON FOR IMBALANCED DATA\n")
cat("==========================================\n")

# Create a copy of the original split data to avoid modifying the main workflow
train_orig <- train_data
test_orig <- test_data

# Analyze class imbalance
class_counts <- table(train_orig$Target_Var)
imbalance_ratio <- class_counts[1] / class_counts[2]
cat("Class distribution in training data:\n")
print(class_counts)
cat("Imbalance ratio (majority:minority):", round(imbalance_ratio, 2), ":1\n\n")

# Create function to evaluate models using consistent metrics
evaluate_model <- function(model, test_data, name = "Model") {
  # Make sure we're working with a compatible test dataset
  # First check if model can make predictions on this test data
  tryCatch({
    pred_prob <- predict(model, newdata = test_data, type = "response")
    
    # Ensure pred_prob has the same length as the test data target
    if(length(pred_prob) != length(test_data$Target_Var)) {
      stop("Prediction length doesn't match test data length")
    }
    
    # Calculate ROC and find optimal threshold
    roc_obj <- roc(test_data$Target_Var, pred_prob)
    auc_val <- auc(roc_obj)
    
    # Get optimal threshold based on Youden's J statistic
    # Wrap in tryCatch to handle potential errors
    optimal_threshold <- tryCatch({
      threshold <- coords(roc_obj, "best", best.method = "youden")[1]
      # Ensure it's numeric
      if(is.numeric(threshold)) {
        threshold
      } else {
        NA_real_
      }
    }, error = function(e) {
      cat("Error getting optimal threshold:", e$message, "\n")
      NA_real_
    })
    
    # Create confusion matrix with optimal threshold
    pred_class <- ifelse(pred_prob > optimal_threshold, 1, 0)
    
    # Convert to factor with same levels if needed
    if(is.factor(test_data$Target_Var)) {
      pred_class <- factor(pred_class, levels = levels(test_data$Target_Var))
    }
    
    # Calculate metrics manually to avoid confusion matrix issues
    accuracy <- mean(pred_class == test_data$Target_Var)
    
    # Handle edge cases for precision and recall
    if(sum(pred_class == 1) == 0) {
      precision <- 0
    } else {
      precision <- sum(pred_class == 1 & test_data$Target_Var == 1) / sum(pred_class == 1)
    }
    
    recall <- sum(pred_class == 1 & test_data$Target_Var == 1) / sum(test_data$Target_Var == 1)
    
    if(precision == 0 && recall == 0) {
      f1 <- 0
    } else {
      f1 <- 2 * precision * recall / (precision + recall)
    }
    
    specificity <- sum(pred_class == 0 & test_data$Target_Var == 0) / sum(test_data$Target_Var == 0)
    balanced_acc <- (recall + specificity) / 2
    
    # Return as data frame
    return(data.frame(
      Model = name,
      Accuracy = accuracy,
      Precision = precision,
      Recall = recall,
      Specificity = specificity,
      F1_Score = f1,
      BalancedAccuracy = balanced_acc,
      AUC = auc_val,
      OptimalThreshold = optimal_threshold
    ))
  }, error = function(e) {
    # If something goes wrong, print the error and return NA values
    cat("Error in model evaluation for", name, ":", e$message, "\n")
    return(data.frame(
      Model = name,
      Accuracy = NA,
      Precision = NA,
      Recall = NA,
      Specificity = NA,
      F1_Score = NA,
      BalancedAccuracy = NA,
      AUC = NA,
      OptimalThreshold = NA
    ))
  })
}

# 1. Original imbalanced data (reference model)
cat("1. Training logistic regression on original imbalanced data...\n")
formula_str <- paste("Target_Var ~", paste(predictors, collapse = " + "))
original_model <- glm(
  as.formula(formula_str),
  family = binomial(link = "logit"),
  data = train_orig
)
original_perf <- evaluate_model(original_model, test_orig, "Original")

# 2. Downsampling the majority class
cat("2. Training logistic regression with downsampling...\n")
set.seed(123)
down_train <- downSample(
  x = train_orig[, !names(train_orig) %in% "Target_Var"],
  y = train_orig$Target_Var,
  yname = "Target_Var"
)
down_model <- glm(
  as.formula(formula_str),
  family = binomial(link = "logit"),
  data = down_train
)
down_perf <- evaluate_model(down_model, test_orig, "Downsampling")

# 3. Upsampling the minority class
cat("3. Training logistic regression with upsampling...\n")
set.seed(123)
up_train <- upSample(
  x = train_orig[, !names(train_orig) %in% "Target_Var"],
  y = train_orig$Target_Var,
  yname = "Target_Var"
)
up_model <- glm(
  as.formula(formula_str),
  family = binomial(link = "logit"),
  data = up_train
)
up_perf <- evaluate_model(up_model, test_orig, "Upsampling")

# Try a simpler approach for SMOTE implementation using ovun.sample from ROSE
if(!require(ROSE)) {
  install.packages("ROSE")
  library(ROSE)
}

set.seed(123)
# Use ovun.sample from ROSE which is more reliable
smote_data <- ovun.sample(Target_Var ~ ., data = train_orig, 
                          method = "both",  # both over and under sampling
                          p = 0.5,         # target 50% of each class
                          seed = 123)$data

cat("SMOTE/balanced data class distribution: ", table(smote_data$Target_Var)[1], "vs", 
    table(smote_data$Target_Var)[2], "\n")

# Train model with SMOTE-balanced data
smote_model <- glm(
  as.formula(formula_str),
  family = binomial(link = "logit"),
  data = smote_data
)
smote_perf <- evaluate_model(smote_model, test_orig, "SMOTE")

# Combine all performance metrics
all_results <- rbind(original_perf, down_perf, up_perf, smote_perf)
cat("\nPerformance comparison across sampling methods:\n")
print(all_results)

# Create visualization of results
sampling_results_long <- melt(all_results, 
                              id.vars = "Model", 
                              measure.vars = c("Accuracy", "Precision", "Recall", 
                                               "Specificity", "F1_Score", "BalancedAccuracy", "AUC"),
                              variable.name = "Metric", 
                              value.name = "Value")

# Fix NA values for visualization
sampling_results_long$Value[is.na(sampling_results_long$Value)] <- 0

# Create bar chart comparing metrics across models
png("sampling_methods_comparison.png", width = 1200, height = 800, res = 100)
ggplot(sampling_results_long, aes(x = Model, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  facet_wrap(~ Metric, scales = "free") +
  scale_fill_brewer(palette = "Set1") +
  geom_text(aes(label = sprintf("%.3f", Value)), position = position_stack(vjust = 0.5)) +
  labs(title = "Comparison of Sampling Methods for Imbalanced Data",
       subtitle = paste("Class imbalance ratio:", round(imbalance_ratio, 2), ":1"),
       x = "Sampling Method", 
       y = "Value") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
dev.off()

# Create ROC curve comparison
png("sampling_methods_roc_curves.png", width = 1000, height = 800, res = 100)
# Set up empty plot
roc_original <- try(roc(test_orig$Target_Var, predict(original_model, newdata = test_orig, type = "response")), silent = TRUE)
if(!inherits(roc_original, "try-error")) {
  plot(roc_original, col = "black", lwd = 2, main = "ROC Curves for Different Sampling Methods")
  
  # Add other curves if they can be created
  roc_down <- try(roc(test_orig$Target_Var, predict(down_model, newdata = test_orig, type = "response")), silent = TRUE)
  if(!inherits(roc_down, "try-error")) {
    lines(roc_down, col = "red", lwd = 2)
  }
  
  roc_up <- try(roc(test_orig$Target_Var, predict(up_model, newdata = test_orig, type = "response")), silent = TRUE)
  if(!inherits(roc_up, "try-error")) {
    lines(roc_up, col = "blue", lwd = 2)
  }
  
  roc_smote <- try(roc(test_orig$Target_Var, predict(smote_model, newdata = test_orig, type = "response")), silent = TRUE)
  if(!inherits(roc_smote, "try-error")) {
    lines(roc_smote, col = "green", lwd = 2)
  }
  
  # Create legend with only the models that worked
  legend_text <- c()
  legend_cols <- c()
  
  if(!inherits(roc_original, "try-error")) {
    legend_text <- c(legend_text, paste("Original (AUC =", round(auc(roc_original), 3), ")"))
    legend_cols <- c(legend_cols, "black")
  }
  
  if(!inherits(roc_down, "try-error")) {
    legend_text <- c(legend_text, paste("Downsampling (AUC =", round(auc(roc_down), 3), ")"))
    legend_cols <- c(legend_cols, "red")
  }
  
  if(!inherits(roc_up, "try-error")) {
    legend_text <- c(legend_text, paste("Upsampling (AUC =", round(auc(roc_up), 3), ")"))
    legend_cols <- c(legend_cols, "blue")
  }
  
  if(!inherits(roc_smote, "try-error")) {
    legend_text <- c(legend_text, paste("SMOTE (AUC =", round(auc(roc_smote), 3), ")"))
    legend_cols <- c(legend_cols, "green")
  }
  
  # Add legend
  legend("bottomright", legend = legend_text, col = legend_cols, lwd = 2)
} else {
  # If we can't create any ROC curves, create a simple plot with a message
  plot(0, 0, type = "n", main = "ROC Curve Generation Failed", 
       xlab = "False Positive Rate", ylab = "True Positive Rate",
       xlim = c(0,1), ylim = c(0,1))
  text(0.5, 0.5, "Unable to generate ROC curves due to errors in model predictions")
}
dev.off()

# Identify best method based on balanced accuracy
# Add safety check for NA values
all_results_filtered <- all_results[!is.na(all_results$BalancedAccuracy),]
if(nrow(all_results_filtered) > 0) {
  best_method <- all_results_filtered[which.max(all_results_filtered$BalancedAccuracy), "Model"]
} else {
  best_method <- "Original" # Default if all are NA
}

# Same safety check for F1
all_results_filtered <- all_results[!is.na(all_results$F1_Score),]
if(nrow(all_results_filtered) > 0) {
  best_f1 <- all_results_filtered[which.max(all_results_filtered$F1_Score), "Model"]
} else {
  best_f1 <- "Original" # Default if all are NA
}

cat("\nSAMPLING METHODS COMPARISON SUMMARY\n")
cat("==========================================\n")
cat("Best method based on Balanced Accuracy:", best_method, "\n")
cat("Best method based on F1 Score:", best_f1, "\n\n")
cat("Class imbalance ratio in original data:", round(imbalance_ratio, 2), ":1\n\n")

cat("Sampling methods evaluation:\n")
for (method in all_results$Model) {
  row <- all_results[all_results$Model == method,]
  cat(paste0("- ", method, ":\n"))
  cat(paste0("  Balanced Accuracy: ", round(row$BalancedAccuracy, 4), "\n"))
  cat(paste0("  F1 Score: ", round(row$F1_Score, 4), "\n"))
  cat(paste0("  AUC: ", round(row$AUC, 4), "\n"))
  
  # Check if threshold is numeric before formatting
  if(!is.null(row$OptimalThreshold) && !is.na(row$OptimalThreshold) && is.numeric(row$OptimalThreshold)) {
    cat(paste0("  Optimal Threshold: ", round(row$OptimalThreshold, 2), "\n\n"))
  } else {
    cat("  Optimal Threshold: NA\n\n")
  }
}

cat("RECOMMENDATIONS:\n")
cat(paste0("Based on our analysis, the ", best_method, " approach provides the best balance\n"))
cat("between identifying positive and negative cases for logistic regression.\n")
cat("This can be compared with the Random Forest results where SMOTE performed best.\n\n")

# Compare coefficient differences between original and best model
if (!is.null(best_method) && !is.na(best_method) && best_method != "Original") {
  best_model <- switch(best_method,
                       "Downsampling" = down_model,
                       "Upsampling" = up_model,
                       "SMOTE" = smote_model,
                       original_model) # Fallback
  
  orig_coeffs <- coef(summary(original_model))
  best_coeffs <- coef(summary(best_model))
  
  # Compare only coefficients present in both models
  common_vars <- intersect(rownames(orig_coeffs), rownames(best_coeffs))
  
  if (length(common_vars) > 0) {
    coeff_comparison <- data.frame(
      Variable = common_vars,
      Original = orig_coeffs[common_vars, 1],
      Best = best_coeffs[common_vars, 1],
      Difference = best_coeffs[common_vars, 1] - orig_coeffs[common_vars, 1],
      PercentChange = (best_coeffs[common_vars, 1] - orig_coeffs[common_vars, 1]) / 
        abs(orig_coeffs[common_vars, 1]) * 100
    )
    
    # Sort by absolute percent change
    coeff_comparison <- coeff_comparison[order(abs(coeff_comparison$PercentChange), decreasing = TRUE),]
    
    cat("COEFFICIENT CHANGES WITH SAMPLING METHOD\n")
    cat("The following variables showed the largest changes in coefficient values:\n")
    print(head(coeff_comparison, 10))
    
    # Visualize coefficient changes
    png("coefficient_changes_with_sampling.png", width = 1000, height = 800, res = 100)
    top_vars <- head(coeff_comparison, 10)$Variable
    plot_data <- coeff_comparison[coeff_comparison$Variable %in% top_vars,]
    
    plot_data_long <- melt(plot_data[, c("Variable", "Original", "Best")], 
                           id.vars = "Variable", 
                           variable.name = "Model", 
                           value.name = "Coefficient")
    
    ggplot(plot_data_long, aes(x = reorder(Variable, Coefficient), y = Coefficient, fill = Model)) +
      geom_bar(stat = "identity", position = "dodge") +
      coord_flip() +
      scale_fill_manual(values = c("steelblue", "firebrick")) +
      labs(title = paste("Coefficient Changes: Original vs", best_method),
           subtitle = "Top 10 variables with largest relative changes",
           x = "Variable", 
           y = "Coefficient Value") +
      theme_minimal()
    dev.off()
  }
}

# Compare model performance to the stepwise model from section 7
if(exists("performance_05")) {
  cat("\nCOMPARISON WITH STEPWISE MODEL\n")
  cat("==========================================\n")
  cat("Stepwise Model (threshold=0.5):\n")
  cat(paste0("  Accuracy: ", round(performance_05$Value[1], 4), "\n"))
  cat(paste0("  Precision: ", round(performance_05$Value[2], 4), "\n"))
  cat(paste0("  Recall: ", round(performance_05$Value[3], 4), "\n"))
  cat(paste0("  Specificity: ", round(performance_05$Value[4], 4), "\n"))
  cat(paste0("  F1 Score: ", round(performance_05$Value[5], 4), "\n\n"))
  
  # Get best model metrics
  best_model_metrics <- all_results[all_results$Model == best_method,]
  
  cat("Best Sampling Method (", best_method, "):\n", sep="")
  cat(paste0("  Accuracy: ", round(best_model_metrics$Accuracy, 4), "\n"))
  cat(paste0("  Precision: ", round(best_model_metrics$Precision, 4), "\n"))
  cat(paste0("  Recall: ", round(best_model_metrics$Recall, 4), "\n"))
  cat(paste0("  Specificity: ", round(best_model_metrics$Specificity, 4), "\n"))
  cat(paste0("  F1 Score: ", round(best_model_metrics$F1_Score, 4), "\n"))
  cat(paste0("  AUC: ", round(best_model_metrics$AUC, 4), "\n"))
  cat(paste0("  Optimal Threshold: ", round(best_model_metrics$OptimalThreshold, 2), "\n\n"))
  
  # Calculate percentage improvements
  recall_improvement <- (best_model_metrics$Recall - performance_05$Value[3]) / performance_05$Value[3] * 100
  f1_improvement <- (best_model_metrics$F1_Score - performance_05$Value[5]) / performance_05$Value[5] * 100
  
  cat("Improvements with", best_method, "sampling:\n")
  cat(paste0("  Recall improvement: ", round(recall_improvement, 1), "%\n"))
  cat(paste0("  F1 Score improvement: ", round(f1_improvement, 1), "%\n\n"))
}

cat("\nFINAL RECOMMENDATIONS FOR HANDLING CLASS IMBALANCE\n")
cat("==========================================\n")
cat("1. ", best_method, " performed best for logistic regression in this analysis.\n", sep="")
cat("2. Threshold optimization is also important - the optimal threshold was found to be ", 
    round(all_results[all_results$Model == best_method, "OptimalThreshold"], 2), 
    " rather than the default 0.5.\n", sep="")
cat("3. For production deployment, we recommend using both ", best_method, 
    " and threshold optimization to maximize model performance.\n", sep="")
cat("4. Coefficient interpretations should be based on the ", best_method, 
    " model to ensure they account for class imbalance.\n", sep="")
if(exists("performance_05")) {
  cat("5. This approach improved recall by ", round(recall_improvement, 1), 
      "% and F1 score by ", round(f1_improvement, 1), "% compared to the original stepwise model.\n", sep="")
}

cat("\nAnalysis complete. Visualizations saved.\n")