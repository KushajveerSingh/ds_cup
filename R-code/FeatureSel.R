library(RCurl)
x <- getURL("https://raw.githubusercontent.com/KushajveerSingh/ds_cup/main/processed_data/tr_bal.csv")
dat <- read.csv(text = x)
dat <- dat[,1:20]

# Scaling to 0 and 1
range01 <- function(x){(x-min(x))/(max(x)-min(x))}
df = apply(dat,2,range01)

# creating all combinations
Mycomb <- function(elements, simplify = FALSE){
  result <- lapply(seq_along(elements), function(m)
    combn(elements, m, simplify = simplify))
  
  result
}

# running Mycomb function
combinations <- Mycomb(1:20)
sum(lengths(Mycomb(1:19)))

# creating a list of sub datasets with all combinations
biglist <- lapply(unlist(combinations, recursive = FALSE), function(k) df[,k])
sub_df_list <- biglist[unlist(lapply(biglist, function(x) "Default_ind" %in% colnames(x)))]

# running univariate LR and compute BIC for each model
# WARNING: It involves 524287 models calculation. Make Sure You have enough memory to run
errors <- sapply(
  sub_df_list,
  function(v) AIC(glm(Default_ind ~ ., v, family = 'binomial'),k = log(3172))
)

# returning the best featyures with the lowest BIC
colnames(as.data.frame(sub_df_list[which.min(errors)]))


