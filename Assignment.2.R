"
########################################################
##################### Assignment 2 #####################
########################################################
Assignment
Dataset: “MI.RData” 
About: 
- Patients that enter hospital with a myocardial infarction. 
- The dataset includes information on the patient at the acceptance in the hospital after the first evaluation. 
- This includes the medical history of the patient, past cardiac problems for example, as well as immediate measurements on the patient, including blood pressure and blood analysis. 
- The response variable (LET_IS) is death of the patient during their stay. 
- In other words, we try to predict from our initial data if the patient is at an elevated risk to die during their hospital stay.
"

setwd("~/School_work/Statistical Methods/Project2")
load("MI.RData")

"
1.
- Study and describe the predictor variables. 
- Do you see any issues that are relevant for making predictions? 
- Discuss the dimensionality of the data and the implication on fitting models.
"

# Check the structure of the dataframe
str(MI)
summary(MI)
# Dimensionality 
dim(MI)
# row:1700  column:101

# Check for missing data
colSums(is.na(MI))

# Remove rows with Missing Data 
MI.clean <- na.omit(MI) # cases: 1700 -> 1695

# Get frequency table
table(MI.clean$LET_IS)

# Check for 'Multicollinearity' using VIF
glm.MI1 <- glm(LET_IS~., data=MI.clean, family = binomial)
# aliases <- alias(lm.MI) # find aliases (linearly dependent terms)
summary(glm.MI1)
# Check VIF values for collinearity inspection 
library(car)
vif_vals <- vif(glm.MI1)
# Identify high Variance Inflation Factor (VIF) values (VIF > 10)
high_vif <- names(vif_vals[vif_vals > 10])
print(high_vif)

# Remove predictors with high VIF (Collinear predictors)
MI.clean <- MI.clean[, !(names(MI.clean) %in% high_vif)]
dim(MI.clean) # variables: 100 -> 94

# Heatmap to visualize correlation between features
library(ggcorrplot)
# Calculate correlation matrix (excluding AGE and SEX)
corr_matrix <- cor(MI[,4:101], use="pairwise")
ggcorrplot(corr_matrix, type="lower", colors=c('blue','white','red'))
# OR using 'corrplot'
library(corrplot)
corrplot(corr_matrix, method = "color")


"
2. 
- Fit and compare an appropriate unconstrained linear model, as well as lasso and ridge regression models. 
- Discuss what you find. What is an appropriate base-level of performance to compare your models to?
"
set.seed(1) # for reproducibility
library(glmnet) # alpha=0: ridge ; alpha=1: lasso

# Create model matrix for predictors
x <- model.matrix(LET_IS ~ ., MI.clean)[, -1]
y <- as.numeric(MI.clean$LET_IS) - 1 # Convert factor to binary numeric (0, 1)

# Split data into training and testing sets 
train_index <- sample(1 : nrow(x), nrow(x) * 0.8)
x.train <- x[train_index, ]
y.train <- y[train_index]
x.test <- x[-train_index, ]
y.test <- y[-train_index]
MI.train <- MI.clean[train_index,]
MI.test <- MI.clean[-train_index,]


## Fit a logistic regression on training set
glm.MI <- glm(LET_IS ~ ., data = MI.train, family=binomial)
summary(glm.MI)
coef.glm <- coef(glm.MI)

## Fit Lasso regularized logistic model
lasso.MI <- cv.glmnet(x.train, y.train, familiy = "binomial", alpha = 1) 
plot(lasso.MI) # plot MSE by lambda value
lasso.coef <- coef(lasso.MI)

## Fit Ridge regularized logistic regression
ridge.MI <- cv.glmnet(x.train, y.train, familiy = "binomial", alpha = 0) 
plot(ridge.MI)
ridge.coef <- coef(ridge.MI) 

## Compare the performances of linear logistic, lasso, and ridge models using AUC
# Generate predictions of response variable using Test data
linear_pred <- predict(glm.MI, newdata = MI.test, type = "response") # compute the probabilities by selecting: type = "response"
lasso_pred <- predict(lasso.MI, newx = x.test, type = "response") 
ridge_pred <- predict(ridge.MI, newx = x.test, type = "response")

plot(linear_pred ~ MI.test$AGE)

# Ensure predictions are numeric vectors
linear_pred <- as.vector(linear_pred)
lasso_pred <- as.vector(lasso_pred)
ridge_pred <- as.vector(ridge_pred)
class(lasso_pred)

# Verify lengths
length(y.test)
length(linear_pred)
length(lasso_pred)
length(ridge_pred)

# Compute ROC curves
library(pROC)
roc_linear <- roc(y.test, linear_pred)
roc_lasso <- roc(y.test, lasso_pred)
roc_ridge <- roc(y.test, ridge_pred)

# Plot ROC curves: y-axis represent Sensitivity; True Positive Rate (TPR), x-axis represents Specificity; False Positive Rate (FPR)
par(mfrow=c(1,1))
plot(roc_linear, col = "blue", main = "ROC Curves", xlab="False positive rate\n(1 - specificity)", ylab = "True Positive Rate (Sensitivity)")
plot(roc_lasso, col = "red", add = TRUE)
plot(roc_ridge, col = "green", add = TRUE)
legend("bottomright", legend = c("Linear", "Lasso", "Ridge"), col = c("blue", "red", "green"), lty = 1)

# Evaluate performance
cat("AUC Linear: ", auc(roc_linear), "\n")
cat("AUC Lasso: ", auc(roc_lasso), "\n")
cat("AUC Ridge: ", auc(roc_ridge), "\n")
# Unconstrained linear model has the highest AUC.

"
3. 
- Among your top predictors, do you see evidence of non-linear effects?
- How could you accommodate non-linear effects and still use a regularized regression approach? 
- Does adding non-linear effects improve your model?
"
# Check for Non-Linear Effects (Non-linearity)

# Top predictor from linear model based on p-value
top_predictors <- names(sort(summary(glm.MI)$coefficients[, "Pr(>|z|)"]))[1:10]
# Top predictor from lasso model based on coefficient(≠0)
top_predictors.lasso <- rownames(lasso.coef)[which(lasso.coef!=0)]

# Top predictors for instance: AGE, STENOK_AN, NA_KB

par(mfrow=c(2,2))
# Function to plot predictor against response
plot_predic_resp <- function(predictor) {
  ggplot(MI.test, aes_string(x = predictor, y = "LET_IS")) +
    geom_jitter(alpha = 0.1) +                      # make dots visible with jitters
    geom_smooth(method = "loess", color = "blue") + # add Loess line
    labs(title = paste("Predictor:", predictor), x = predictor, y = "LET_IS")
}
# Loop through top predictors and print plots
for (predictor in top_predictors) {
  print(plot_predic_resp(predictor))
}

## Residual Plot to check non-linearity
# Get Fitted values for linear model
fitted_val.lm <- fitted(glm.MI)
# Get Standardized residuals

rs.lm <- rstandard(glm.MI)
# Plot Residual against Fitted values
par(mfrow = c(1, 2))
plot(rs.lm~fitted_val.lm, main="Residual vs Fitted value", ylab = "Standard Residuals") # Clearly not random pattern 
abline(h=0, lty=2)

# Residual against each top predictor
plot(rs.lm~MI.train$AGE, main="AGE", ylab = "Standard Residuals") # Clearly not random pattern 
abline(h=0, lty=2)
plot(rs.lm~MI.train$STENOK_AN , main="STENOK_AN", ylab = "Standard Residuals") 
plot(rs.lm~MI.train$NA_KB , main="NA_KB", ylab = "Standard Residuals") 

# Partial residual (Component+Residual) plots for the GLM model
library(car)
crPlots(glm.MI)

# fitted value vs predictors 'AGE' (test data)
plot(linear_pred~MI.test$AGE) 

## Polynomial Regression
# Create polynomial terms 
library(ggplot2)
library(lattice)
library(caret)

# Create polynomial features for training and test sets
x_train_poly_2 <- polynomialFeatures(x.train, degree = 2)
poly_feats <- preProcess(MI.clean[,-1], method = "poly", degree = 2, pca = FALSE)
MI.poly <- predict(poly_feats, MI.clean[, -1])

## Splines
library(splines)


# Fit Lasso-regularized logistic model with spline features
lasso_spline <- cv.glmnet(x.train_spline, y.train, family = "binomial", alpha = 1)
plot(lasso_spline)
coef(lasso_spline, s = "lambda.min")

# Fit Ridge-regularized logistic regression with spline features
ridge_spline <- cv.glmnet(x.train_spline, y.train, family = "binomial", alpha = 0)
plot(ridge_spline)
coef(ridge_spline, s = "lambda.min")

# Fit Regularized Models with Non-linear terms 
# Create polynomial terms for each predictor
library(caret)
poly_features <- preProcess(MI.clean[, -1], method = "poly", degree = 2, pca = FALSE)
MI.poly <- predict(poly_features, MI.clean[, -1])

# Combine with the response variable
MI.poly <- cbind(MI.poly, LET_IS = MI.clean$LET_IS)

# GAM

"
4. 
- Fit an appropriate Random Forest model. 
- Report a comparison of performance to your linear model and explain any differences in performance. 
- Do you see an important difference in how variables are used for predictions?
"
library(randomForest)
rf_model <- randomForest(LET_IS ~ ., data=MI.train, mtry=sqrt(ncol(MI.train) - 1), importance=TRUE) 
print(rf_model)

# Examine the feature importance of variables in the Random Forest model.
importance(rf_model)
varImpPlot(rf_model)

# Generate prediction with rf model on test data
rf_pred <- predict(rf_model, newdata = MI.test, type="prob")[,2]
as.vector(MI.test$LET_IS)
roc_rf <- roc(as.numeric(MI.test$LET_IS)-1, rf_pred)

# Plot ROC curve
par(mfrow=c(1,1))
plot(roc_rf, col="green", main="Random Forest", xlab = "False positive rate\n(1 - specificity)")
plot(roc_linear, col = "blue", main = "Linear", add = TRUE)
plot(roc_lasso, col = "red", main = "Lasso", add = TRUE)
legend("bottomright", legend = c("Random Forest", "Linear", "Lasso"), col = c("green", "blue", "red"), lty = 1)

# Evaluate performance
cat("AUC Random Forest: ", auc(roc_rf), "\n") # 0.9094811 
cat("AUC Linear: ", auc(roc_linear), "\n") # 0.8634441 
cat("AUC Lasso: ", auc(roc_lasso), "\n") # 0.8220729 


