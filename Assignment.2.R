"
########################################################
##################### Assignment 2 #####################
########################################################
Assignment
Dataset: “MI” 
About: 
- Patients that enter hospital with a myocardial infarction. 
- The dataset includes information on the patient at the acceptance in the hospital after the first evaluation. 
- This includes the medical history of the patient, past cardiac problems for example, as well as immediate measurements on the patient, including blood pressure and blood analysis. 
- The response variable (LET_IS) is death of the patient during their stay. 
- In other words, we try to predict from our initial data if the patient is at an elevated risk to die during their hospital stay.
"
MI = read.csv(file = file.choose(), header=T)
dim(MI)
head(MI) 
"
1. 
Study and describe the predictor variables. 
Do you see any issues that are relevant for making predictions? 
Make sure to discuss the dimensionality of the data and the implication on fitting models.
"
str(MI)
summary(MI)

# Check for missing data
colSums(is.na(MI))

# Dimensionality 
cor(MI, use = "complete.obs")
"
2. Fit and compare an appropriate unconstrained linear model, as well as lasso and ridge regression models. 
   Discuss what you find. What is an appropriate base-level of performance to com- pare your models to?
"
is.factor(MI$LET_IS)
MI$LET_IS <- as.factor(MI$LET_IS)
lm.MI <- glm(LET_IS ~ ., data=MI, family=binomial)

