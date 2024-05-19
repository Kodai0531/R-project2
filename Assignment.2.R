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

# Check for missing data
colSums(is.na(MI))

# Function to calculate the proportion of missing values
pMiss <- function(x){sum(is.na(x))/length(x)}
apply(MI, 2, pMiss)

# check if response var. is a factor
is.factor(MI$LET_IS)

# Dimensionality 
dim(MI)

# check for multicollinearity
library(car)
aliases <- alias(lm.MI)
print(aliases)

vif_vals <- vif(lm.MI)
print(vif_vals)


high_vif <- 

"
# Discuss missing data , multicollinearity, and high dimensionality
# 100 predictor variables(dimensions)
Discuss Dimensionality:
- Number of predictors vs. number of observations.
- Potential multicollinearity issues.
- Implications for model fitting (e.g., overfitting).
"

"
2. 
- Fit and compare an appropriate unconstrained linear model, as well as lasso and ridge regression models. 
- Discuss what you find. What is an appropriate base-level of performance to compare your models to?
"

# remove rows with Missing Data 
MI.clean <- na.omit(MI)
dim(MI.clean)

# Cross-validation
set.seed(1) # for reproducibility

# Create model matrix for predictors
x <- model.matrix(LET_IS ~ ., MI.clean)[, -1]

# Create response variable
y <- as.numeric(MI.clean$LET_IS) - 1 # Convert factor to binary numeric (0, 1)

# Check dimensions
dim(x)
length(y)
# Ensured: the same # of row and observations

library(glmnet)
# alpha=0: ridge ; alpha=1: lasso

# Split data into training and testing sets (train:test=7:3)
train_index <- sample(1 : nrow(x), nrow(x) * 0.7)
x.train <- x[train_index, ]
y.train <- y[train_index]
x.test <- x[-train_index, ]
y.test <- y[-train_index]

## Fit a logistic regression on training set
lm.MI <- glm(LET_IS ~ ., data = MI.clean[train_index,], family=binomial)
summary(lm.MI)


## Fit Lasso-regularized logistic model
lasso.MI <- cv.glmnet(x.train, y.train, familiy = "binomial", alpha = 1) 
plot(lasso.MI) # plot MSE by lambda value
bestlm.lasso <- lasso.MI$lambda.min # extract coefficients for the best(smallest) lambda (s: lambda)
coef(lasso.MI, s = "lambda.min")

## Fit Ridge-regularized logistic regression
ridge.MI <- cv.glmnet(x.train, y.train, familiy = "binomial", alpha = 0) 
plot(ridge.MI)
bestlm.ridge <- ridge.MI$lambda.min
coef(ridge.MI, s = "lambda.min") 


## Compare the performances of linear logistic, lasso, and ridge models using AUC
# Generate predictions of "response" variable
linear_pred <- predict(lm.MI, newdata = MI.clean[-train_index, ], type = "response")
lasso_pred <- predict(lasso.MI, s = bestlm.lasso, newx = x.test, type = "response") 
ridge_pred <- predict(ridge.MI, s = bestlm.ridge, newx = x.test, type = "response")

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
plot(roc_linear, col = "blue", main = "ROC Curves", xlab="False positive rate\n(1 - specificity)", ylab = "True Positive Rate (Sensitivity)")
plot(roc_lasso, col = "red", add = TRUE)
plot(roc_ridge, col = "green", add = TRUE)
legend("bottomright", legend = c("Linear", "Lasso", "Ridge"), col = c("blue", "red", "green"), lty = 1)

"
3. 
- Among your top predictors, do you see evidence of non-linear effects?
- How could you accommodate non-linear effects and still use a regularized regression approach? 
- Does adding non-linear effects improve your model?
"
# Check for Non-Linear Effects
top_predictors <- names(sort(abs(coef(lasso.MI, s = "lambda.min")), decreasing = TRUE))
pairs(MI.clean[, c(top_predictors, "LET_IS")])
install.packages("Matrix")
install.packages('carData')
library(car)
# Create partial residual plots for each predictor
crPlots(lm.MI)
# Identify top predictors


# Check for linearity: Residual Plot
# Get fitted values
fitted_val.lm <- fitted(lm.MI)
# Get standardized residuals
rs.lm <- rstandard(lm.MI)
# shapiro.test(standardized_rs) # p-value < 2.2e-16 (Normality check)

# Residual against fitted values
par(mfrow = c(1, 2))
plot(rs.lm~fitted_val.lm, ylab = "Standard Residuals") # Clearly not random pattern 
# Residual against each regressor(independent variable)
plot(rs.lm~MI[train_index,]$AGE , ylab = "Standard Residuals") # Clearly not random pattern 
abline(0, 0, lty = 2)

"
Residual vs Fitted values
The plot exhibited the non-random scatter pattern, which indicate non-linear relationships between the predictors and the response variables.

Residual vs AGE
(The other plot of shows Residual against one of the top predictors, 'AGE'.)
The plot also exhibited non-random scatter pattern, which also support non-linearity

Also, the spread of residuals varies with the fitted values, suggesting 'Heteroscedasticity' (the variance of the residuals is not constant). 

It is reasonable to  include non-linear terms (e.g., polynomial terms, splines) or use a more flexible model like a Generalized Additive Model(GAM).
"



"
4. 
- Fit an appropriate Random Forest model. 
- Report a comparison of performance to your linear model and explain any differences in performance. 
- Do you see an important difference in how variables are used for predictions?
"




