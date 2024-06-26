library(randomForest)
library(gam)
library(tree)
library(MASS)
library(glmnet)#lasso and ridge
library(splines)
library(ggplot2)
library(cowplot)
library(pROC)


##1. Study and describe the predictor variables. 
#Do you see any issues that are relevant for making predictions? 
#Make sure to discuss the dimensionality of the data and the implication on fitting models.

load("MI.RData")
dim(MI) #it appears enough data for the number of variables
table(MI[,1]) # make frequency table; but response has imbalance; hence firmly in data scarcity for the contrast
#The models will therefore easily suffer from high variance and we will need to take this into account.
# Identify which variables are factors
which(sapply(MI, is.factor))
table(MI[,c(1,3)]) #response distribution is okish over both sexes

# Density plots
my_plots <- lapply(names(MI), function(var_x){
  p <- ggplot(MI) +aes(.data[[var_x]])
  if(is.numeric(MI[[var_x]])) {
    p <- p + geom_density()
  } else {
    p <- p + geom_bar()
  } 
})

summary(MI)
# Save density and bar plots to PDF
pdf(paste("distributions.pdf",sep=""),height=100,width=100)
# Arrange multiple plots into a single layout or grid.
plot_grid(plotlist = my_plots) # cowplot package
dev.off() # close current graphics device in R (pdf file device)

#boxplots
my_plots <- lapply(names(MI), function(var_x){
  p <- ggplot(MI) +aes(.data[[var_x]])
  if(is.numeric(MI[[var_x]])) {
    p <- p + geom_boxplot(outlier.colour="black", outlier.shape=16,
                          outlier.size=2, notch=FALSE)+coord_flip()
  } else {
    p <- p + geom_bar()
  } 
})
 
pdf(paste("boxplots.pdf",sep=""),height=100,width=100)
plot_grid(plotlist = my_plots)
dev.off()

# Many of the variables have very irregular distributions; some variables are semi-categorical, some (also) are v. unbalanced

# Examine correlations between the subset of more continuous variables
cont=c(1,2,35,36,82,84,85,86,87,88,89) # vector of cols for pair plots 
pdf(paste("pairs.pdf",sep=""),height=100,width=100)
pairs(MI[,cont]) 
dev.off()
# S_AD_ORIT and D_AD_ORIT are correlated, same for K and Na-blood, and ALT and AST-blood

# Missing data
which(is.na(MI))#NA values
MI[!complete.cases(MI), ] #some values missing for 5 variables for the same 5 rows

set.seed(5)
# there are few affected observations, so dropping them is possible
# but let's avoid loosing one TRUE response and do imputation
y=as.factor(MI$LET_IS)
x=MI[,-1]
x2=rfImpute(x,y,iter=5,ntree=300)[,-1] # one approach to infer values


##2. Fit and compare an appropriate unconstrained linear model, as well as lasso and ridge regression models.
#Discuss what you find. 
#What is an appropriate base-level of performance to compare your models to?

#we use a training/test set to compare the models
set.seed(5)
# Split data into training and testing sets (2/3 training, 1/3 testing)
train=sample(1:nrow(MI), nrow(MI)*2/3) 
test=(-train)

# due to the imbalance in the response, the base level performance is not simply 50/50.
# As a base-level of performance can be taken a simple classifier with no/minimal learning (always FALSE for example) or a nearest neighbor approach. 

# Base-level performance using a simple classifier (only intercept):Empty(null) model >> provide 'Baseline performance'
model0=glm(y~1,data=x2,subset=train,family="binomial") #only intercept, simply choosing majority in effect
predEmpty = predict(model0,newdata=x2[test,], type="response")
res=predEmpty>0.5 # converts probabilities into binary outcomes(predictions). TRUE(1) if >0.5, FALSE(0) if <0.5
table(y[test],res) # confusion matrix: counts true and predicted class labels: the performance with True positive, False negative, etc.
# y[test]: True vals; res: predicted vals

#I use the accuracy as a performance measure here, as it is the simplest one. 
performanceEmpty=length(which(res==y[test]))/length(y[test]) # Accuracy: Number of Correct predictions/Total number of Observations
performanceEmpty # the ratio of correct predictions out of the total number of predictions.

#Other measures: Area Under the Receiver Operating Curve (AUC)
roc_obj <- roc(y[test], predEmpty)
plot(roc_obj,print.auc=TRUE)
auc(roc_obj)

#Fit and compare an -appropriate- unconstrained linear model, as well as 'lasso' and 'ridge' regression models. 
model3=glm(y~., data=x2, subset=train, family="binomial") #full unconstrained model
#glm.fit: fitted probabilities numerically 0 or 1 occurred
predFull = predict(model3,newdata=x2[test,],type="response")
res=predFull>0.5
table(y[test],res) 
performanceFull=length(which(res==y[test]))/length(y[test])
performanceFull #Worse than just the intercept! High variance component in the error, most likely.

#The warning messages point to problems with this data set and fitting the model, 
#most likely related to limited observations for some predictor and response values

# Forward/Backward stepwise selection approach to refine the model.
# Greedy fwd (Adding one by one) with AIC, more stringent bwd with Chisq tests
# why AIC and Chisq?: AIC > useful for adding var; Chi-sq > useful for individual predictor's contribution.
library(MASS)
model4=stepAIC(model0,direction="forward",scope=list(upper=model3,lower=model0)) # scope: define upper(full)/lower(just intercept) models
c=coefficients(summary(model4))
# fwd why once? >> selection stops once no additional predictor significantly lowers the AIC
# Forward selection identifies the most promising variables for inclusion
# Next, additional refinement with backward selection by removing less significant predictors.

# Backwards selection with Chisq test: removing one by one from the model fit in fwd.
model5=update(model4, . ~ . -GT_POST )
# anova(complex, simpler) check if simpler model is significantly worse (complex model is better aka adding the param improve the model) 
anova(model4,model5,test='Chisq') # not to drop
summary(model5)
model6=update(model5, . ~ . -ROE )
anova(model5,model6,test='Chisq') # significant > not
summary(model6)
model7=update(model6, . ~ . -n_p_ecg_p_01 )
anova(model6,model7,test='Chisq') # significant > not
summary(model7)
model8=update(model7, . ~ . -n_r_ecg_p_03 )
anova(model7,model8,test='Chisq') # not to drop
summary(model8)
model9=update(model8, . ~ . -zab_leg_01 )
anova(model8,model9,test='Chisq') # significant > not
summary(model9)
model10=update(model9, . ~ . -NOT_NA_KB )
anova(model9,model10,test='Chisq') # drop
summary(model10)
model11=update(model10, . ~ . -np08 )
anova(model10,model11,test='Chisq') # not to drop
summary(model11)
model12=update(model11, . ~ . -IBS_NASL )
anova(model11,model12,test='Chisq') # drop
summary(model12)
model13=update(model12, . ~ . -np10 )
anova(model12,model13,test='Chisq') # not to drop
summary(model13) 
# can't be dropped since doing so doesn't change the model deviance and degree of freedom, indicating the parameter is essential or the models are identical in terms of fit.
coefficients(summary(model13))

# Normal GLM regression with minimal selected variables
predGLM = predict(model12,newdata=x2[test,],type="response")
res=predGLM>0.5
table(y[test],res) 
performanceGLM=length(which(res==y[test]))/length(y[test]) 
#just give the AUC of ROC once more
roc_obj <- roc(y[test], predGLM)
plot(roc_obj,print.auc=TRUE)
auc(roc_obj)

# Fit and compare 'lasso' and 'ridge' regression models
set.seed(10)
grid =10^seq (0,-5, length =100) #Defines a sequence of lambda values (regularization parameters) to be tested.
modelLASSO =cv.glmnet(x=as.matrix(x2[train,-1]),y[train], alpha =1, lambda =grid,family="binomial")
par(mfrow=c(1,1))
plot(modelLASSO)#the effect of high variance with complete models is v.clear
modelLASSO.bestLambda = modelLASSO$lambda.min
modelLASSO.pred = predict(modelLASSO,s=modelLASSO.bestLambda, newx=as.matrix(x2[test,-1]),type="response")
resLasso=modelLASSO.pred>0.5
table(y[test],resLasso)
performanceLASSO=length(which(resLasso==y[test]))/length(y[test])
# Extract coeff from lasso model
vals=predict(modelLASSO,s=modelLASSO.bestLambda,type="coefficients")

# ridge
set.seed(10)
modelRIDGE =cv.glmnet(x=as.matrix(x2[train,]),y[train],alpha =0, lambda =grid,family="binomial")
plot(modelRIDGE)#a more regular curve than Lasso
modelRIDGE.bestLambda = modelRIDGE$lambda.1se #min is also OK here
modelRIDGE.pred = predict(modelRIDGE,s=modelRIDGE.bestLambda, newx=as.matrix(x2[test,]),type="response")
resRidge=modelRIDGE.pred>0.5
table(y[test],resRidge) 
performanceRidge=length(which(resRidge==y[test]))/length(y[test])
vals=predict(modelRIDGE,s=modelRIDGE.bestLambda,type="coefficients")
#just the UAROC again
roc_obj <- roc(y[test], modelRIDGE.pred[,1])
plot(roc_obj,print.auc=TRUE)
auc(roc_obj)

performanceEmpty
performanceFull
performanceGLM
performanceRidge
performanceLASSO
# the linear models show modest performance effects

##3. Among your 'top predictors', do you see evidence of non-linear effects? 
#How could you accommodate non-linear effects and still use a regularized regression approach? 
#Does adding non-linear effects improve your model?

#lets look at the LASSO selected values
vals
# fit a GAM to assess the non-linear effect for top-predictor: KFK_BLOOD (why?)
modelGAM1=gam(y~s(KFK_BLOOD,4), data=x2, subset=train, family="binomial")
# s(): smooth term for KFK_BLOOD with 4 degree of freedom
summary(modelGAM1)
# highly significant p value in 'Non-parametric effect': non-linear component of 'KFK_BLOOD' significantly improved
# >> strong non-linear effect
plot(modelGAM1)

#the correlated D_AD_ORIT and S_AD_ORIT are also interesting
modelGAM2=gam(y~s(D_AD_ORIT,4)+s(S_AD_ORIT,4),data=x2[,],subset=train,family="binomial")
summary(modelGAM2) # also both showed non-linear effects
par(mfrow=c(2,1))
plot(modelGAM2)
par(mfrow=c(1,1))

# Let's take the matrix with these selected coefficients and account for non-linear effects in two ways
# Prepare Data for regularized model with non-linear terms
# create new dataset 'x3' incorporating non-linear terms for the top predictors.
x3=x2[,as.array(!vals==0)] # select vars with non-zero coefficients from Lasso
#first add some polynomials for KFK_BLOOD
x3$KFK_BLOOD2=x3$KFK_BLOOD*x3$KFK_BLOOD # Square term for 'KFK_BLOOD'
x3$KFK_BLOOD3=x3$KFK_BLOOD2*x3$KFK_BLOOD # Cubic term for 'KFK_BLOOD'

# We can also cheat a bit and use modelGAM2 to recode the variables
# Adds a new var predicted by 'modelGAM2' capturing non-linear effects of 'D_AD_ORIT' and 'S_AD_ORIT'.
x3$AD_ORIT_GAM=predict(modelGAM2,newdata=x2,type="response") 

# Fit Ridge Regression with Non-Linear Effects by using new dataset x3
modelRIDGE_NLeffects =cv.glmnet(x=as.matrix(x3[train,]),y[train],alpha =0, lambda =grid,family="binomial")
plot(modelRIDGE_NLeffects)#a more regular curve than Lasso

modelRIDGE_NLeffects.bestLambda = modelRIDGE_NLeffects$lambda.min 
modelRIDGE_NLeffects.pred = predict(modelRIDGE_NLeffects,s=modelRIDGE_NLeffects.bestLambda, newx=as.matrix(x3[test,]),type="response")
resRidge_nl=modelRIDGE_NLeffects.pred>0.5
table(y[test],resRidge_nl)
# calculate the performance
performanceRidge_NLeffects=length(which(resRidge_nl==y[test]))/length(y[test])
vals2=predict(modelRIDGE_NLeffects,s=modelRIDGE.bestLambda,type="coefficients")
roc_obj <- roc(y[test], modelRIDGE_NLeffects.pred[,1])
plot(roc_obj,print.auc=TRUE)
auc(roc_obj)
#so this is a tiny bit better! 

#4. Fit an appropriate Random Forest model. 
# Report a comparison of performance to your linear model and explain any differences in performance. 
# Do you see an important difference in how variables are used for predictions?

rf=randomForest(y=y[train],x=x2[train,],mtry=(sqrt(ncol(x))),importance=TRUE,ntree=1000,nodesize=20) 
# mtry: # of randomly sampled as candidates at each split; importance: variable importance is calculated; nodesize: min size of terminal nodes
predRF = predict(rf,newdata=x2[test,]) 
# Evaluate performance
table(y[test],predRF) 
performanceRF=length(which(predRF==y[test]))/length(y[test])
performanceRF #best up until now
# Variable importance
imp=importance(rf)
varImpPlot(rf)
# ROC Curve and AUC
roc_obj <- roc(y[test], modelRIDGE_NLeffects.pred[,1])
plot(roc_obj,print.auc=TRUE)
auc(roc_obj) #same as for the nl-ridge

# Compare LASSO and Random Forest Variable importance
# let's see the variables selected by LASSO vs their RF importance scores
# Matches the variable names from the importance table to those from the Lasso model coefficients.
i=match(rownames(imp),rownames(vals))
plot(as.factor(!vals[i]==0),imp[,4])
#Clearly related, but not perfectly. Some values have reasonable scores, but are not in the LASSO model
#We can look at the p-values of the log. glm model we fit earlier. 
pvals=coef(summary(model12))[-1,4]
i2=match(names(pvals),rownames(imp))
plot(log(pvals),imp[i2,4])
summary(lm(log(pvals)~imp[i2,4]))

#a very modest relation! 
#The two modeling approaches relate the predictors v. differently to the response
#The trees are much more apt at dealing with the near categorical data we have hear
#further non-linear effects and interactions between variables are natively incorporated. 
