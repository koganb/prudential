# ensure the results are repeatable
set.seed(7)
# load the library
library(mlbench)
library(caret)
# load the data


train <- read.csv("data/train.csv",stringsAsFactors = T) #59,381 observations, 128 variables

# calculate correlation matrix
correlationMatrix <- cor(train[, c(paste("Medical_Keyword_",1:48,sep=""),'Response')])

# summarize the correlation matrix
print(correlationMatrix)

# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.5)
# print indexes of highly correlated attributes
print(highlyCorrelated)