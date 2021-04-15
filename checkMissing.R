library(RCurl)
# read training data with missing values
x <- getURL("https://raw.githubusercontent.com/KushajveerSingh/ds_cup/main/processed_data/train.csv")
df.tr<- read.csv(text = x)
################################################ check misshing values
# check randomness in uti_card_50plus_pct
ml1 <- glm(Default_ind~-1+isNaN_uti_card_50plus_pct,family = 'binomial', data = df.tr)
# check randomness in rep_income
ml2 <- glm(Default_ind~-1+isNaN_rep_income,family = 'binomial', data = df.tr)
# check randomness in bith rep_income and uti_card_50plus_pct
ml3 <- glm(Default_ind~-1+isNaN_rep_income+isNaN_uti_card_50plus_pct,family = 'binomial', data = df.tr)
summary(ml1)
summary(ml2)
summary(ml3)

# check rare cases
View(df.tr[df.tr$avg_card_debt>90000,c(1,2,5,8,11,12,13,20)])
