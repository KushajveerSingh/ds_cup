library(RCurl)
# read test data with the output predictions of fair rf model
x <- getURL("https://raw.githubusercontent.com/KushajveerSingh/ds_cup/main/processed_data/rf_fair_orig.csv")
df.mod<- read.csv(text = x)
################################################ aggregation bias for xyz bank
library('lmtest')
library(dplyr)
library(emmeans)

count<-function(x){sum(x==0)}

df.mod$bin_uti_card = ntile(df.mod$uti_card,5)

# prepare data for analyzing aggregation bias
xyzbank <- aggregate( df.mod[,c("class_pred_fair")], df.mod[,c("bin_uti_card","ind_acc_XYZ")], 
                      FUN = count )
xyzbank$bad <- aggregate( df.mod[,c("class_pred_fair")], df.mod[,c("bin_uti_card","ind_acc_XYZ")], 
                          FUN = sum )[[3]]
colnames(xyzbank)[c(3,4)] <- c("good","bad")
xyzbank$bin_uti_card = as.factor(xyzbank$bin_uti_card)
xyzbank$ind_acc_XYZ = as.factor(xyzbank$ind_acc_XYZ)

# Model Building Approach
# Sum to zero constraints
contrasts(xyzbank$bin_uti_card) <- contr.sum(5, contrasts=TRUE)
contrasts(xyzbank$ind_acc_XYZ) <- contr.sum(2, contrasts=TRUE)

# GLM with ind_acc_XYZ
ml1 <- glm(cbind(good, bad) ~ ind_acc_XYZ, 
           data = xyzbank,
           family = binomial)

# adding bin_uti_card for controlling factor
ml2a <- update(ml1, . ~ . + bin_uti_card)


summary(ml1)
summary(ml2a) # bin 5 in uti_card = 1.64972 (sum to zero effect)
AIC(ml2a) # returns AIC.

# Test Nested Models
lrtest(ml1, ml2a)

# Plot Probability of Acceptance by xyz & bin_uti_card
emmip(ml2a, ind_acc_XYZ ~ bin_uti_card, type = "response", CI = TRUE)
