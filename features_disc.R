#####################################################################################
### This script is broken into 5 parts and provides example code to:
#   1) Read in the competition data, concatenate the train and test data, define variables as numeric or factor for the gbm
#   2) Create some simple features which can be used in a predictive model
#   3) Recreate train and test now that features have been created on both
#   4) Build a simple GBM on a random 70% of train, validate on the other 30% and calculate the quadratic weighted kappa
#   5) Score the test data and create a submission file
##############################################
###################################################################################

library("futile.logger")
flog.threshold(DEBUG)



train <- read.csv(file=file.path("./data", "train.csv"),stringsAsFactors = T) #59,381 observations, 128 variables
test <- read.csv(file=file.path("./data", "test.csv"),stringsAsFactors = T)   #19,765 observations, 127 variables - test does not have a response field


res <- apply(train, 2, function(col)sum(is.na(col))/length(col))
res[res>0]
length(res[res>0])






