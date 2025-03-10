###############################################################################
# LOGISTIC REGRESSION WITH LOCATION RISK SCORING, THRESHOLD TUNING, 
# CATEGORICAL HANDLING, AND DETAILED INTERPRETATION
###############################################################################
# This script demonstrates a full workflow:
# 1) Import and prepare data (including factor conversion).
# 2) Transform S_LEGAL into a numeric location risk score.
# 3) Remove highly correlated numeric variables via caret::findCorrelation.
# 4) Improve categorical variable handling based on actual counts.
# 5) Build a logistic regression model using dynamic formula construction.
# 6) Perform stepwise selection.
# 7) Visualize results, compare thresholds (0.3 vs. 0.5).
# 8) Provide a detailed interpretive summary of the model and variable effects.
# 9) Create a focused visualization of significant coefficients.
# 10) Create Python-style feature importance visualizations.
# 11) Compare original and improved models.

# Load required packages
if(!require(corrplot)) {
  install.packages("corrplot")
  library(corrplot)
}
if(!require(dplyr)) {
  install.packages("dplyr")
  library(dplyr)
}
if(!require(ggplot2)) {
  install.packages("ggplot2")
  library(ggplot2)
}
if(!require(caret)) {
  install.packages("caret")
  library(caret)
}
if(!require(car)) {
  install.packages("car")
  library(car)
}
if(!require(pROC)) {
  install.packages("pROC")
  library(pROC)
}
if(!require(gridExtra)) {
  install.packages("gridExtra")
  library(gridExtra)
}
if(!require(reshape2)) {
  install.packages("reshape2")
  library(reshape2)
}

###############################################################################
# 1) DATA IMPORT AND PREPARATION
###############################################################################
# Set file path (adjust as needed)
raw_path <- r"(C:\Users\User\Desktop\Job Search\USF_School_Project\Integrative_Program_Project_Cohort_2022_MSBAIS\Model_Optimization_in_R\INPUTS.csv)"

# Import data (this now contains more columns than before)
# Set na.strings parameter to include empty strings
# Import data and replace empty strings with 0
data_full <- read.csv(raw_path, stringsAsFactors = FALSE)

# Replace empty strings with 0 throughout the entire dataframe
data_full[data_full == ""] <- 0


# Select only the columns needed for the original workflow
data <- data_full %>% 
  select(LND_SQFOOT, TOT_LVG_AREA, S_LEGAL, CONST_CLASS, IMP_QUAL, JV, 
         LND_VAL, NO_BULDNG, NCONST_VAL, DEL_VAL, SPEC_FEAT_VAL, 
         MonthDifference, SALE_PRC1, Target_Var, EFF_AGE, ACT_AGE)

# Convert target variable to factor (binary classification)
data$Target_Var <- as.factor(data$Target_Var)

# Because "0" in CONST_CLASS and IMP_QUAL is actually invalid, rename it to "NULL" so your later code
# will group it as UNKNOWN:
data$IMP_QUAL[data$IMP_QUAL == "0"] <- "NULL"
data$CONST_CLASS[data$CONST_CLASS == "0"] <- "NULL"

# Display the first few rows to verify the data
head(data)

# Store original data for later comparison
original_data <- data
###############################################################################
# 2) IMPROVED CATEGORICAL VARIABLE HANDLING BASED ON COUNTS
###############################################################################
cat("Improving categorical variable handling based on actual counts...\n")

# Print initial distributions
cat("\nOriginal IMP_QUAL distribution:\n")
table_imp_qual <- table(data$IMP_QUAL, useNA = "ifany")
print(table_imp_qual)

cat("\nOriginal CONST_CLASS distribution:\n")
table_const_class <- table(data$CONST_CLASS, useNA = "ifany")
print(table_const_class)

# IMP_QUAL handling improvements: based on counts from actual data
# IMP_QUAL counts: 3 (30.8M), 4 (12.4M), 2 (7.1M), 5 (2.6M), 1 (1.7M), NULL (651K), 6 (431K), 7 (2K)
# Strategy: Group rare categories (6,7) and handle NULL values
data$IMP_QUAL_MOD <- data$IMP_QUAL
data$IMP_QUAL_MOD[data$IMP_QUAL_MOD %in% c("6", "7")] <- "6_7" # Combine rare values 6 and 7
data$IMP_QUAL_MOD[is.na(data$IMP_QUAL_MOD) | data$IMP_QUAL_MOD == "NULL"] <- "UNKNOWN" # Handle NULL/NA

# CONST_CLASS handling improvements: based on counts from actual data
# CONST_CLASS counts: 3 (23M), NULL (19.2M), 4 (8.6M), 2 (4.9M), 1 (69K), 5 (50K)
# Strategy: Group rare categories (1,5) and handle NULL values
data$CONST_CLASS_MOD <- data$CONST_CLASS
data$CONST_CLASS_MOD[data$CONST_CLASS_MOD %in% c("1", "5")] <- "1_5" # Combine rare values 1 and 5
data$CONST_CLASS_MOD[is.na(data$CONST_CLASS_MOD) | data$CONST_CLASS_MOD == "NULL"] <- "UNKNOWN" # Handle NULL/NA
# Set level "3" as the reference level for CONST_CLASS_MOD
data$CONST_CLASS_MOD <- relevel(factor(data$CONST_CLASS_MOD), ref = "3")

# Convert modified categorical variables to proper factors with explicit levels
data$IMP_QUAL_MOD <- factor(data$IMP_QUAL_MOD)
# Set level "3" as the reference level for IMP_QUAL_MOD
data$IMP_QUAL_MOD <- relevel(factor(data$IMP_QUAL_MOD), ref = "3")
data$CONST_CLASS_MOD <- factor(data$CONST_CLASS_MOD)

# Check the new distributions
cat("\nModified IMP_QUAL_MOD distribution:\n")
table_imp_qual_mod <- table(data$IMP_QUAL_MOD, useNA = "ifany")
print(table_imp_qual_mod)

cat("\nModified CONST_CLASS_MOD distribution:\n")
table_const_class_mod <- table(data$CONST_CLASS_MOD, useNA = "ifany")
print(table_const_class_mod)

# Create bar plots of the original vs. modified categorical distributions
png("categorical_improvements_imp_qual.png", width = 1000, height = 600, res = 100)
par(mfrow = c(1, 2))
barplot(table_imp_qual, main = "Original IMP_QUAL Distribution", 
        ylab = "Count", xlab = "IMP_QUAL", col = "skyblue", log = "y")
barplot(table_imp_qual_mod, main = "Modified IMP_QUAL Distribution", 
        ylab = "Count", xlab = "IMP_QUAL_MOD", col = "steelblue", log = "y")
dev.off()

png("categorical_improvements_const_class.png", width = 1000, height = 600, res = 100)
par(mfrow = c(1, 2))
barplot(table_const_class, main = "Original CONST_CLASS Distribution", 
        ylab = "Count", xlab = "CONST_CLASS", col = "lightgreen", log = "y")
barplot(table_const_class_mod, main = "Modified CONST_CLASS Distribution", 
        ylab = "Count", xlab = "CONST_CLASS_MOD", col = "darkgreen", log = "y")
dev.off()

###############################################################################
# 3) TRANSFORM S_LEGAL INTO LOCATION RISK SCORE
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

# Create all model data including location_risk but excluding original variables
# that have been replaced with improved versions
model_data <- data[, !names(data) %in% c("S_LEGAL", "IMP_QUAL", "CONST_CLASS")]

# Create a histogram of location risk to understand its distribution
png("location_risk_distribution.png", width = 800, height = 600, res = 100)
hist(data$location_risk, 
     main = "Distribution of Location Risk Scores",
     xlab = "Risk Score (probability of loss)",
     col = "steelblue",
     breaks = 30)
abline(v = avg_risk, col = "red", lwd = 2, lty = 2)
text(avg_risk + 0.02, max(hist(data$location_risk, plot = FALSE)$counts) * 0.8, 
     paste("Average:", round(avg_risk, 3)), col = "red")
dev.off()

###############################################################################
# 3.5) OUTLIER TRIMMING BASED ON IQR
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
numeric_cols_for_outliers <- setdiff(numeric_cols_for_outliers, "Target_Var") # Exclude Target_Var

# Create a dataframe to track outliers per column
outlier_counts <- data.frame(
  Variable = numeric_cols_for_outliers,
  OutlierCount = 0,
  OutlierPercent = 0
)

# Identify outliers for each column
outlier_matrix <- matrix(FALSE, nrow = nrow(model_data), ncol = length(numeric_cols_for_outliers))
colnames(outlier_matrix) <- numeric_cols_for_outliers

for (i in 1:length(numeric_cols_for_outliers)) {
  col <- numeric_cols_for_outliers[i]
  outlier_matrix[,i] <- identify_outliers(model_data[[col]])
  outlier_counts$OutlierCount[i] <- sum(outlier_matrix[,i], na.rm = TRUE)
  outlier_counts$OutlierPercent[i] <- outlier_counts$OutlierCount[i] / nrow(model_data) * 100
}

# Print outlier summary
outlier_counts <- outlier_counts[order(outlier_counts$OutlierCount, decreasing = TRUE),]
cat("Outlier summary by variable:\n")
print(outlier_counts)

# Flag rows with multiple outliers
outlier_row_counts <- rowSums(outlier_matrix, na.rm = TRUE)
multi_outlier_rows <- outlier_row_counts >= 2

cat("Rows with multiple outliers:", sum(multi_outlier_rows), "out of", nrow(model_data), 
    "(", round(sum(multi_outlier_rows)/nrow(model_data) * 100, 2), "%)\n")

# Remove rows with multiple outliers
model_data_clean <- model_data[!multi_outlier_rows,]
cat("Data dimensions after removing multi-outlier rows:", dim(model_data_clean)[1], "x", dim(model_data_clean)[2], "\n")

# Create a histogram to see outlier distribution by row
png("outlier_distribution.png", width = 800, height = 600, res = 100)
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

###############################################################################
# 4) CORRELATION ANALYSIS ON NUMERIC VARIABLES
###############################################################################
# We only compute correlation among numeric columns. Factor columns are excluded.
numeric_cols <- names(model_data)[sapply(model_data, is.numeric)]
model_data_numeric <- model_data[numeric_cols]

# Temporarily convert Target_Var to numeric for correlation, if present
if ("Target_Var" %in% numeric_cols) {
  model_data_numeric$Target_Var <- as.numeric(as.character(model_data_numeric$Target_Var))
}

# Generate correlation matrix and visualize it
cor_matrix <- cor(model_data_numeric, use = "complete.obs")
png("correlation_heatmap.png", width = 1000, height = 800, res = 100)
corrplot(cor_matrix, method = "color", type = "upper",
         tl.col = "black", tl.srt = 45,
         addCoef.col = "black", number.cex = 0.7,
         title = "Correlation Matrix of Numeric Variables")
dev.off()

# Check correlation of location_risk with other variables
loc_risk_cors <- cor_matrix["location_risk", ]
loc_risk_cors <- loc_risk_cors[order(abs(loc_risk_cors), decreasing = TRUE)]
cat("\nCorrelations with location_risk:\n")
print(round(head(loc_risk_cors, 10), 4))

# Automatically remove variables with correlation > 0.7
remove_indices <- findCorrelation(cor_matrix, cutoff = 0.7, verbose = TRUE)
remove_vars <- colnames(model_data_numeric)[remove_indices]
cat("Variables to remove due to high correlation:", paste(remove_vars, collapse = ", "), "\n")

# Create optimized dataset by excluding flagged numeric variables
optimized_data <- model_data[, !names(model_data) %in% remove_vars]

###############################################################################
# 5) SPLIT INTO TRAIN/TEST WITH STRATIFIED SAMPLING
###############################################################################
set.seed(123)

# Use stratified sampling to ensure representative distribution of categories
# This is especially important for the imbalanced categorical variables
cat("\nImplementing stratified sampling for train/test split...\n")

# Create stratification variable that combines Target_Var and key categorical variables
strat_var <- with(optimized_data, paste(Target_Var, IMP_QUAL_MOD, CONST_CLASS_MOD))

# Use createDataPartition from caret for stratified sampling
train_index <- createDataPartition(strat_var, p = 0.7, list = FALSE)
train_data <- optimized_data[train_index, ]
test_data <- optimized_data[-train_index, ]

# Verify stratification worked
cat("\nTraining data class distribution:\n")
print(table(train_data$Target_Var))
cat("\nTest data class distribution:\n")
print(table(test_data$Target_Var))

cat("\nTraining data IMP_QUAL_MOD distribution:\n")
print(prop.table(table(train_data$IMP_QUAL_MOD)))
cat("\nTest data IMP_QUAL_MOD distribution:\n")
print(prop.table(table(test_data$IMP_QUAL_MOD)))

###############################################################################
# 6) BUILD DYNAMIC FORMULA AND STEPWISE SELECTION
###############################################################################
# Build formula adaptively, excluding Target_Var
predictors <- setdiff(names(train_data), "Target_Var")
formula_str <- paste("Target_Var ~", paste(predictors, collapse = " + "))
cat("Adaptive Model Formula:\n", formula_str, "\n")

# Fit full logistic model
full_model <- glm(
  as.formula(formula_str),
  family = binomial(link = "logit"),
  data = train_data
)

# Perform stepwise selection (both directions)
step_model <- step(full_model, direction = "both", trace = 0)  # Set trace=0 to reduce output verbosity

# Display stepwise model summary
summary_step <- summary(step_model)
cat("\nFinal Stepwise Model Summary:\n")
print(summary_step)

# Check VIF for remaining multicollinearity issues
vif_vals <- try(vif(step_model), silent = TRUE)
if(!inherits(vif_vals, "try-error")) {
  cat("\nVariance Inflation Factors (VIF):\n")
  print(vif_vals)
  cat("VIF > 5 may indicate problematic multicollinearity.\n")
}

###############################################################################
# 7) THRESHOLD TUNING AND PERFORMANCE EVALUATION
###############################################################################
pred_prob <- predict(step_model, newdata = test_data, type = "response")

# Compare performance at threshold=0.3 vs. threshold=0.5

### Threshold = 0.3
threshold_1 <- 0.3
pred_class_03 <- ifelse(pred_prob > threshold_1, 1, 0)
conf_03 <- table(pred_class_03, test_data$Target_Var)
accuracy_03 <- sum(diag(conf_03)) / sum(conf_03)
precision_03 <- conf_03[2,2] / sum(conf_03[2,])
recall_03 <- conf_03[2,2] / sum(conf_03[,2])
f1_03 <- 2 * precision_03 * recall_03 / (precision_03 + recall_03)
specificity_03 <- conf_03[1,1] / sum(conf_03[,1])

performance_03 <- data.frame(
  Metric = c("Accuracy", "Precision", "Recall/Sensitivity", "Specificity", "F1 Score"),
  Value  = c(accuracy_03, precision_03, recall_03, specificity_03, f1_03)
)

### Threshold = 0.5
threshold_2 <- 0.5
pred_class_05 <- ifelse(pred_prob > threshold_2, 1, 0)
conf_05 <- table(pred_class_05, test_data$Target_Var)
accuracy_05 <- sum(diag(conf_05)) / sum(conf_05)
precision_05 <- conf_05[2,2] / sum(conf_05[2,])
recall_05 <- conf_05[2,2] / sum(conf_05[,2])
f1_05 <- 2 * precision_05 * recall_05 / (precision_05 + recall_05)
specificity_05 <- conf_05[1,1] / sum(conf_05[,1])

performance_05 <- data.frame(
  Metric = c("Accuracy", "Precision", "Recall/Sensitivity", "Specificity", "F1 Score"),
  Value  = c(accuracy_05, precision_05, recall_05, specificity_05, f1_05)
)

cat("\nPerformance at threshold =", threshold_1, ":\n")
print(performance_03)
cat("\nPerformance at threshold =", threshold_2, ":\n")
print(performance_05)

# Compare AUC
roc_obj <- roc(test_data$Target_Var, pred_prob)
auc_value <- auc(roc_obj)
cat("\nAUC for Stepwise Model:", round(auc_value, 4), "\n")

# Plot ROC Curve
png("roc_curve_thresholds.png", width = 800, height = 800, res = 100)
plot(roc_obj, main = paste("ROC Curve (AUC =", round(auc_value, 3), ")"))
dev.off()

###############################################################################
# 8) FIND OPTIMAL THRESHOLD
###############################################################################
# Calculate metrics across different thresholds
thresholds <- seq(0.1, 0.9, by = 0.05)
threshold_results <- data.frame(
  Threshold = thresholds,
  Accuracy = numeric(length(thresholds)),
  Precision = numeric(length(thresholds)),
  Recall = numeric(length(thresholds)),
  F1_Score = numeric(length(thresholds)),
  Specificity = numeric(length(thresholds))
)

for (i in 1:length(thresholds)) {
  thresh <- thresholds[i]
  pred_class <- ifelse(pred_prob > thresh, 1, 0)
  conf <- table(pred_class, test_data$Target_Var)
  
  # Handle the case where the confusion matrix doesn't have all classes
  if (nrow(conf) == 1) {
    # If only one class is predicted
    if (rownames(conf)[1] == "0") {
      # Only 0s predicted
      threshold_results$Accuracy[i] <- conf[1,1] / sum(conf)
      threshold_results$Precision[i] <- NA
      threshold_results$Recall[i] <- 0
      threshold_results$F1_Score[i] <- 0
      threshold_results$Specificity[i] <- 1
    } else {
      # Only 1s predicted
      threshold_results$Accuracy[i] <- conf[1,2] / sum(conf)
      threshold_results$Precision[i] <- conf[1,2] / sum(conf[1,])
      threshold_results$Recall[i] <- 1
      threshold_results$F1_Score[i] <- 2 * threshold_results$Precision[i] / (threshold_results$Precision[i] + 1)
      threshold_results$Specificity[i] <- 0
    }
  } else {
    # Normal case with 2x2 confusion matrix
    threshold_results$Accuracy[i] <- sum(diag(conf)) / sum(conf)
    threshold_results$Precision[i] <- conf[2,2] / sum(conf[2,])
    threshold_results$Recall[i] <- conf[2,2] / sum(conf[,2])
    threshold_results$F1_Score[i] <- 2 * threshold_results$Precision[i] * threshold_results$Recall[i] / 
      (threshold_results$Precision[i] + threshold_results$Recall[i])
    threshold_results$Specificity[i] <- conf[1,1] / sum(conf[,1])
  }
}

# Find optimal threshold for F1 score
optimal_f1_idx <- which.max(threshold_results$F1_Score)
optimal_threshold <- threshold_results$Threshold[optimal_f1_idx]

cat("\nOptimal threshold for maximizing F1 Score:", optimal_threshold, "\n")
cat("Performance at optimal threshold:\n")
print(threshold_results[optimal_f1_idx, ])

# Plot threshold analysis
threshold_long <- reshape2::melt(threshold_results, id.vars = "Threshold", 
                                 variable.name = "Metric", value.name = "Value")

png("threshold_optimization.png", width = 1000, height = 600, res = 100)
ggplot(threshold_long, aes(x = Threshold, y = Value, color = Metric)) +
  geom_line(size = 1) +
  geom_point() +
  geom_vline(xintercept = optimal_threshold, linetype = "dashed", color = "black") +
  labs(title = "Model Performance Metrics at Different Thresholds",
       subtitle = paste("Optimal threshold for F1 score:", optimal_threshold),
       x = "Classification Threshold",
       y = "Metric Value") +
  theme_minimal()
dev.off()

###############################################################################
# 9) COEFFICIENTS AND VARIABLE IMPORTANCE INTERPRETATION
###############################################################################
# Extract model coefficients and significance
coeffs <- coef(summary_step)
coeff_data <- data.frame(
  Variable = row.names(coeffs),
  Coefficient = coeffs[,1],
  StdError = coeffs[,2],
  pValue = coeffs[,4],
  Significance = ifelse(coeffs[,4] < 0.001, "***", 
                        ifelse(coeffs[,4] < 0.01, "**",
                               ifelse(coeffs[,4] < 0.05, "*",
                                      ifelse(coeffs[,4] < 0.1, ".", ""))))
)

# Check if location_risk was retained in the model
if("location_risk" %in% coeff_data$Variable) {
  location_coef <- coeff_data[coeff_data$Variable == "location_risk", "Coefficient"]
  location_p <- coeff_data[coeff_data$Variable == "location_risk", "pValue"]
  location_odds <- exp(location_coef)
  cat("\nLocation risk coefficient:", location_coef, "\n")
  cat("Location risk p-value:", location_p, "\n")
  cat("Location risk odds ratio:", location_odds, "\n")
  cat("This means a 10 percentage point increase in location risk is associated with a", 
      round((exp(location_coef * 0.1) - 1) * 100, 2), 
      "% change in odds of loss\n")
}

# Remove intercept, sort by absolute magnitude
coeff_data <- coeff_data[coeff_data$Variable != "(Intercept)", ]
coeff_data <- coeff_data[order(abs(coeff_data$Coefficient), decreasing = TRUE), ]

# Create odds ratios and percentage labels
coeff_data$OddsRatio <- exp(coeff_data$Coefficient)
coeff_data$OddsRatioLabel <- ifelse(
  coeff_data$OddsRatio > 1,
  paste0("+", round((coeff_data$OddsRatio - 1)*100, 1), "%"),
  paste0(round((coeff_data$OddsRatio - 1)*100, 1), "%")
)

# Plot coefficients in descending order (original visualization)
png("optimized_model_coefficients.png", width = 1000, height = 800, res = 100)
ggplot(coeff_data, aes(x = reorder(Variable, Coefficient), y = Coefficient, fill = pValue < 0.05)) +
  geom_bar(stat = "identity", position = "dodge") +
  coord_flip() +
  scale_fill_manual(values = c("gray", "steelblue"), name = "Significant") +
  geom_text(aes(label = OddsRatioLabel),
            hjust = ifelse(coeff_data$Coefficient > 0, -0.1, 1.1),
            size = 3) +
  labs(title = "Optimized Model Coefficients",
       subtitle = "Sorted by absolute magnitude with significance levels",
       x = "",
       y = "Coefficient Value",
       caption = "Significance codes: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1") +
  theme_minimal() +
  theme(legend.position = "bottom")
dev.off()

###############################################################################
# 10) FOCUSED VISUALIZATION OF SIGNIFICANT COEFFICIENTS
###############################################################################
# Filter for significant variables and meaningful effects
significant_coeffs <- coeff_data[coeff_data$pValue < 0.05, ]  # Only significant variables

# Set a threshold for meaningful effect size (e.g., at least a 5% change in odds)
# You can adjust this threshold as needed (0.05 = 5% change)
meaningful_threshold <- 0.05  
meaningful_coeffs <- significant_coeffs[
  abs(significant_coeffs$OddsRatio - 1) >= meaningful_threshold, 
]

# Create a cleaner visualization with only these coefficients
png("significant_model_coefficients.png", width = 1000, height = 700, res = 100)
ggplot(meaningful_coeffs, aes(x = reorder(Variable, Coefficient), y = Coefficient, 
                              fill = Coefficient > 0)) +
  geom_bar(stat = "identity", position = "dodge") +
  coord_flip() +
  scale_fill_manual(values = c("firebrick3", "steelblue3"), 
                    name = "Effect Direction",
                    labels = c("Decreases Loss Risk", "Increases Loss Risk")) +
  geom_text(aes(label = OddsRatioLabel),
            hjust = ifelse(meaningful_coeffs$Coefficient > 0, -0.1, 1.1),
            size = 3.5) +
  labs(title = "Significant Predictors of Property Loss",
       subtitle = paste0("Only showing variables with meaningful effect sizes (>", 
                         meaningful_threshold*100, "% change in odds)"),
       x = "",
       y = "Coefficient Value",
       caption = "Based on logistic regression model with stepwise variable selection and improved category handling") +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank(),
    axis.text.y = element_text(size = 11),
    plot.title = element_text(face = "bold"),
    plot.subtitle = element_text(color = "darkgrey")
  )
dev.off()

cat("\nFocused visualization of significant coefficients created.\n")
cat("Only showing variables with effect size >", meaningful_threshold*100, "% change in odds.\n")
cat("Number of variables in focused visualization:", nrow(meaningful_coeffs), 
    "out of", nrow(significant_coeffs), "significant variables.\n")

# Generate a table of variable importance
var_importance <- data.frame(
  Variable     = coeff_data$Variable,
  AbsCoefficient = abs(coeff_data$Coefficient),
  Coefficient  = coeff_data$Coefficient,
  PValue       = coeff_data$pValue,
  OddsRatio    = coeff_data$OddsRatio,
  Effect       = ifelse(coeff_data$Coefficient > 0,
                        "Increases loss probability",
                        "Decreases loss probability")
)
var_importance <- var_importance[order(var_importance$AbsCoefficient, decreasing = TRUE), ]

cat("\nVariable Importance for Property Loss Prediction:\n")
print(var_importance)

# Practical interpretation: Each numeric predictor's coefficient translates to
# a % change per one-unit increase. Factor levels (like CONST_CLASS_MOD1_5, IMP_QUAL_MOD4)
# compare to a baseline category. 
effects_summary <- data.frame(
  Variable = var_importance$Variable,
  Effect   = ifelse(var_importance$Coefficient > 0,
                    paste0("Increases loss probability by ",
                           round((var_importance$OddsRatio - 1)*100, 1),
                           "% per unit"),
                    paste0("Decreases loss probability by ",
                           abs(round((var_importance$OddsRatio - 1)*100, 1)),
                           "% per unit")),
  Significance = ifelse(var_importance$PValue < 0.05, "Significant", "Not significant")
)

cat("\nPractical Interpretation of Variable Effects:\n")
print(effects_summary)

###############################################################################
# 11) FINAL SUMMARY AND COMMENTS
###############################################################################
cat("\nSUMMARY FOR FINAL MODEL\n")
cat("==========================================\n")

cat("Key variables for predicting property loss:\n\n")
for (i in 1:nrow(effects_summary)) {
  if (effects_summary$Significance[i] == "Significant") {
    cat(paste0("- ", effects_summary$Variable[i], ": ",
               effects_summary$Effect[i], "\n"))
  }
}

cat("\nModel performance metrics at threshold=0.5:\n")
cat(paste0("- Accuracy: ", round(accuracy_05*100, 1), "%\n"))
cat(paste0("- Precision: ", round(precision_05*100, 1), "%\n"))
cat(paste0("- Recall: ", round(recall_05*100, 1), "%\n"))
cat(paste0("- F1 Score: ", round(f1_05*100, 1), "%\n"))
cat(paste0("- AUC: ", round(auc_value, 3), "\n"))

cat("\nModel performance at optimal threshold (", optimal_threshold, "):\n", sep="")
cat(paste0("- Accuracy: ", round(threshold_results$Accuracy[optimal_f1_idx]*100, 1), "%\n"))
cat(paste0("- Precision: ", round(threshold_results$Precision[optimal_f1_idx]*100, 1), "%\n"))
cat(paste0("- Recall: ", round(threshold_results$Recall[optimal_f1_idx]*100, 1), "%\n"))
cat(paste0("- F1 Score: ", round(threshold_results$F1_Score[optimal_f1_idx]*100, 1), "%\n"))

cat("\nThis model addresses multicollinearity by removing highly correlated variables,\n")
cat("incorporates location information through risk scoring rather than categorical variables,\n")
cat("and properly handles imbalanced categorical variables by combining rare categories.\n")

# Create a comparison table between original R and improved R approaches
cat("\nComparison Between Original Python, Basic R, and Improved R Models:\n")
cat("==========================================\n")
cat("Python Model: Used one-hot encoding for S_LEGAL (location) and treated categoricals as numeric\n")
cat("Basic R Model: Used location risk scoring and proper factor handling\n")
cat("Improved R Model: Added intelligent categorical grouping based on frequency counts and stratified sampling\n\n")

cat("Key Enhancements in the Improved R Model:\n")
cat("1. IMP_QUAL handling: Combined rare categories 6 & 7 (only ~433K out of 55M+ records)\n")
cat("2. CONST_CLASS handling: Combined rare categories 1 & 5 (only ~119K out of 55M+ records)\n")
cat("3. NULL value handling: Created explicit 'UNKNOWN' category for missing values\n")
cat("   (Especially important for CONST_CLASS with 19.2M NULL values)\n")
cat("4. Stratified sampling: Ensures rare categories are represented in both training/test sets\n\n")

cat("Overall Model Evolution Summary:\n")
cat("The improved R model builds on the location risk scoring approach that proved superior\n")
cat("to Python's one-hot encoding, while adding sophisticated handling of categorical variables\n")
cat("based on their actual frequency distributions. The stratified sampling further ensures\n")
cat("that model training properly accounts for all category levels despite severe imbalance.\n")

###############################################################################
# 12) PYTHON-STYLE COEFFICIENT VISUALIZATION AND TABLES
###############################################################################

# Function to calculate inv_logit (Python-equivalent of the proportion of loss outcomes)
inv_logit <- function(x) {
  return(exp(x) / (1 + exp(x)))
}

# Create a Python-style coefficient table
python_style_coefs <- data.frame(
  Variable = coeff_data$Variable,
  Coefficient = coeff_data$Coefficient,
  Proportion = inv_logit(coeff_data$Coefficient)
)

# Sort by coefficient (descending)
python_style_coefs <- python_style_coefs[order(python_style_coefs$Coefficient, decreasing = TRUE), ]

# Print the table in Python style
cat("\nPython-Style Coefficient Table:\n")
cat("Predictor Variable\tCoefficient\tProportion of the Loss Outcomes\n")
for (i in 1:nrow(python_style_coefs)) {
  cat(paste0(python_style_coefs$Variable[i], "\t", 
             round(python_style_coefs$Coefficient[i], 3), "\t",
             round(python_style_coefs$Proportion[i], 3), "\n"))
}

# Save the table to CSV file
write.csv(python_style_coefs, "python_style_coefficients.csv", row.names = FALSE)

# Create Python-style feature importance visualization (like yellowbrick FeatureImportances)
# Take top 15 features by absolute coefficient value
top_n <- 15
top_features <- head(coeff_data[order(abs(coeff_data$Coefficient), decreasing = TRUE),], top_n)

# Create a horizontal bar chart in Python-yellowbrick style
png("python_style_feature_importance.png", width = 1000, height = 800, res = 100)
ggplot(top_features, aes(x = reorder(Variable, abs(Coefficient)), y = Coefficient)) +
  geom_bar(stat = "identity", aes(fill = Coefficient > 0)) +
  coord_flip() +
  scale_fill_manual(values = c("#D65F5F", "#4878D0"), 
                    name = "Direction",
                    labels = c("Negative", "Positive")) +
  geom_text(aes(label = sprintf("%.3f", Coefficient)),
            hjust = ifelse(top_features$Coefficient > 0, -0.1, 1.1),
            size = 3.5) +
  labs(title = "Feature Importances (Python-Style Visualization)",
       subtitle = "Top 15 features by coefficient magnitude",
       x = "Features",
       y = "Coefficient Value (Impact on Log-Odds)") +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank(),
    axis.text.y = element_text(size = 11),
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(color = "darkgrey", size = 12),
    panel.background = element_rect(fill = "#f5f5f5", color = NA),
    plot.background = element_rect(fill = "#f5f5f5", color = NA)
  )
dev.off()

# Create heatmap visualization of odds ratios (similar to Python seaborn style)
# Take top 15 features by absolute coefficient
odds_ratio_data <- head(coeff_data[order(abs(coeff_data$Coefficient), decreasing = TRUE), 
                                   c("Variable", "OddsRatio", "pValue")], top_n)

# Compute log odds ratio for better visualization (better scale)
odds_ratio_data$LogOddsRatio <- log(odds_ratio_data$OddsRatio)

# Create a seaborn-style heatmap
png("odds_ratio_heatmap.png", width = 1000, height = 600, res = 100)
ggplot(odds_ratio_data, aes(x = 1, y = reorder(Variable, abs(LogOddsRatio)))) +
  geom_tile(aes(fill = LogOddsRatio), color = "white", size = 0.5) +
  scale_fill_gradient2(low = "#D65F5F", mid = "white", high = "#4878D0", midpoint = 0,
                       name = "Log Odds Ratio") +
  geom_text(aes(label = sprintf("%.2f", OddsRatio)), size = 3.5) +
  labs(title = "Odds Ratios Heatmap",
       subtitle = "Top 15 features by impact magnitude",
       y = "",
       x = "") +
  theme_minimal() +
  theme(
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    panel.grid = element_blank(),
    legend.position = "bottom",
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(color = "darkgrey", size = 12)
  )
dev.off()
