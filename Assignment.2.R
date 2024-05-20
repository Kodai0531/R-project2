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
table(MI$LET_IS)

# Check for missing data
colSums(is.na(MI))

# Dimensionality 
dim(MI)

# Remove rows with Missing Data 
MI.clean <- na.omit(MI)

# Check for 'Multicollinearity' using VIF
glm.MI1 <- glm(LET_IS~., data=MI.clean, family = binomial)
# aliases <- alias(lm.MI) # find aliases (linearly dependent terms)

library(car) 
# Check VIF values
vif_vals <- vif(glm.MI1)
print(vif_vals)

# Identify high Variance Inflation Factor (VIF) values (VIF > 10; as a rule of thumb)
high_vif <- names(vif_vals[vif_vals > 10])
print(high_vif)

# Remove predictors with high VIF (Collinear predictors)
MI.clean <- MI.clean[, !(names(MI.clean) %in% high_vif)]
dim(MI.clean) # 100 vars -> 94 vars

# Heatmap to visualize correlation between features
library(ggcorrplot)
# Calculate correlation matrix (excluding AGE and SEX)
corr_matrix <- cor(MI[,4:101], use="pairwise")
ggcorrplot(corr_matrix, type="lower", colors=c('blue','white','red'))

# OR using 'corrplot'
library(corrplot)
corrplot(corr_matrix, method = "color")

"
# Discuss missing data , multicollinearity, and high dimensionality
# 100 predictor variables(dimensions)
Discuss Dimensionality:
- Number of predictors vs number of observations.
- Potential multicollinearity issues.
- Implications for model fitting (e.g., overfitting).
"

"
2. 
- Fit and compare an appropriate unconstrained linear model, as well as lasso and ridge regression models. 
- Discuss what you find. What is an appropriate base-level of performance to compare your models to?
"

# Cross-validation
set.seed(1) # for reproducibility
library(glmnet) # alpha=0: ridge ; alpha=1: lasso

# Create model matrix for predictors
x <- model.matrix(LET_IS ~ ., MI.clean)[, -1]
y <- as.numeric(MI.clean$LET_IS) - 1 # Convert factor to binary numeric (0, 1)

# Split data into training and testing sets (train:test=7:3)
train_index <- sample(1 : nrow(x), nrow(x) * 0.8)
x.train <- x[train_index, ]
y.train <- y[train_index]
x.test <- x[-train_index, ]
y.test <- y[-train_index]

## Fit a logistic regression on training set, linearizing with "binomial(link = logit)"
glm.MI <- glm(LET_IS ~ ., data = MI.clean[train_index,], family=binomial(link = logit))
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
# Generate predictions of response variable
linear_pred <- predict(glm.MI, newdata = MI.clean[-train_index, ], type = "response") # compute the probabilities by selecting: type = "response"
lasso_pred <- predict(lasso.MI, newx = x.test, type = "response") 
ridge_pred <- predict(ridge.MI, newx = x.test, type = "response")

# plot(linear_pred ~ MI.clean[-train_index,]$AGE)
?predict
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

## Residual Plot
# Get Fitted values for linear model
fitted_val.lm <- fitted(glm.MI)
# Get Standardized residuals
rs.lm <- rstandard(glm.MI)

# Top predictor from glm model based on p-value
names(sort(summary(glm.MI)$coefficients[, "Pr(>|z|)"]))[1:10]
# Top predictors for instance: AGE,STENOK_AN, NA_KB 

# Plot Residual against Fitted values
par(mfrow = c(1, 2))
plot(rs.lm~fitted_val.lm, ylab = "Standard Residuals") # Clearly not random pattern 
abline(h=0, lty=2)

# Residual against each top predictor
plot(rs.lm~MI[train_index,]$AGE, main="AGE", ylab = "Standard Residuals") # Clearly not random pattern 
abline(h=0, lty=2)
plot(rs.lm~MI[train_index,]$STENOK_AN , main="STENOK_AN", ylab = "Standard Residuals") 
plot(rs.lm~MI[train_index,]$NA_KB , main="NA_KB", ylab = "Standard Residuals") 

# Partial residual (Component+Residual) plots for the GLM model
library(car)
crPlots(glm.MI)

# fitted value vs predictors 'AGE' (test data)
plot(linear_pred~MI.clean[-train_index,]$AGE) 



## Polynomial Regression
# Create polynomial terms 

install.packages("caret")
library(ggplot2)
library(lattice)
library(caret)
x_train_poly <- polynomialFeatures(x.train, degree = 2)

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

"
Residual vs Fitted values
The plot exhibited the non-random scatter pattern, which indicate non-linear relationships between the predictors and the response variables.

Residual vs AGE
(The other plot of shows Residual against one of the top predictors, 'AGE'.)
The plot also exhibited non-random scatter pattern, which also support non-linearity

Also, the spread of residuals varies with the fitted values, suggesting 'Heteroscedasticity' (the variance of the residuals is not constant). 

It is reasonable to  include non-linear terms (e.g., polynomial terms, splines) or use a more flexible model like a Generalized Additive Model(GAM).
"

# Fit Regularized Models with Non-linear terms 

# Fit polynomial regression models
glm_poly2 <- glm(LET_IS ~ poly(AGE, 2) + ., data = MI.clean[train_index, ], family = binomial)
glm_poly3 <- glm(LET_IS ~ poly(AGE, 3) + ., data = MI.clean[train_index, ], family = binomial)

# Compare models using AIC
AIC(glm.MI, glm_poly2, glm_poly3)





"
4. 
- Fit an appropriate Random Forest model. 
- Report a comparison of performance to your linear model and explain any differences in performance. 
- Do you see an important difference in how variables are used for predictions?
"
library(randomForest)
rf_model <- randomForest(LET_IS ~ ., data=MI.clean, ntree=500, mtry=sqrt(ncol(MI.clean) - 1))
print(rf_model)

rf_pred <- predict(rf_model, type="prob")[,2]
roc_rf <- roc(y,rf_pred)

# Plot ROC curve
plot(roc_rf, col="blue", main="Random Forest")
plot(roc_linear, col = "gray", main = "Linear", add = TRUE)
plot(roc_lasso, col = "red", main = "Lasso", add = TRUE)

# Examine the feature importance of variables in the Random Forest model.
varImpPlot(rf_model)


