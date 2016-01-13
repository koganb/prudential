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


###############################################################################
#Step 1: Read in the data and define variables as either a factor or numeric
#########################################################################################

train <- read.csv(file=file.path("./data", "train.csv"),stringsAsFactors = T) #59,381 observations, 128 variables
test <- read.csv(file=file.path("./data", "test.csv"),stringsAsFactors = T) #19,765 observations, 127 variables - test does not have a response field

train$Train_Flag <- 1 #Add in a flag to identify if observations fall in train data, 1 train, 0 test
test$Train_Flag <- 0 #Add in a flag to identify if observations fall in train data, 1 train, 0 test
test$Response <- NA #Add in a column for Response in the test data and initialize to NA


#concatenate train and test together, any features we create will be on both data sets with the same code. This will make scoring easy
All_Data <- rbind(train,test) #79,146 observations, 129 variables 



#Define variables as either numeric or factor, Data_1 - Numeric Variables, Data_2 - factor variables
Data_1 <- All_Data[,names(All_Data) %in% c("Product_Info_4",    "Ins_Age",	"Ht",	"Wt",	"BMI",	"Employment_Info_1",	"Employment_Info_4",	"Employment_Info_6",	"Insurance_History_5",	"Family_Hist_2",	"Family_Hist_3",	"Family_Hist_4",	"Family_Hist_5",	"Medical_History_1",	"Medical_History_15",	"Medical_History_24",	"Medical_History_32",paste("Medical_Keyword_",1:48,sep=""))]
Data_2 <- All_Data[,!(names(All_Data) %in% c("Product_Info_4",	"Ins_Age",	"Ht",	"Wt",	"BMI",	"Employment_Info_1",	"Employment_Info_4",	"Employment_Info_6",	"Insurance_History_5",	"Family_Hist_2",	"Family_Hist_3",	"Family_Hist_4",	"Family_Hist_5",	"Medical_History_1",	"Medical_History_15",	"Medical_History_24",	"Medical_History_32",paste("Medical_Keyword_",1:48,sep="")))]
Data_2<- data.frame(apply(Data_2, 2, as.factor))

All_Data <- cbind(Data_1,Data_2) #79,146 observations, 129 variables

#We don't need Data_1,Data_2,train or test anymore
rm(Data_1,Data_2,train,test)

#Look at structure of All_Data - The variables seem to be classed as recommended on the data guide for the competition
#str(All_Data)




##############################################################
#Step 2: Feature Creation - create some features which we want to test in a predictive model
##########################################################



#Make a function which will group variables into buckets based on p, p=0.1 <- deciles. 
#note: the bucket cut offs cannot overlap, thus you are not guaranteed 10 groups with 0.1 nor equally sized groups
group_into_buckets <- function(var,p){
    cut(var, 
        breaks= unique(quantile(var,probs=seq(0,1,by=p), na.rm=T)),
        include.lowest=T, ordered=T) 
}



#Investigate the Wt variable - Normalized weight of applicant

summary(All_Data$Wt)
#     Min.   1st Qu.  Median    Mean      3rd Qu.     Max. 
#    0.0000  0.2259   0.2887   0.2926     0.3452     1.0000


#Make a new variable which is equivalent to the quintile groups of Wt, we can use the group_into_buckets function we defined above

All_Data$Wt_quintile <- group_into_buckets(All_Data$Wt,0.2)
table(All_Data$Wt_quintile)

#[0,0.215]   (0.215,0.268]  (0.268,0.31]  (0.31,0.362]     (0.362,1] 
#   17028         17277         15825         13704          15312

class(All_Data$Wt_quintile)
#"ordered" "factor"



#Investigate the medical keyword fields, would the number of medical keywords equal to 1 have predictive power?

#Function to sum across rows for variables defined
psum <- function(...,na.rm=FALSE) { 
    rowSums(do.call(cbind,list(...)),na.rm=na.rm) }


#Make a new variable which sums across all of the Medical_Keyword dummy variables on an application
All_Data$Number_medical_keywords <- psum(All_Data[,c(paste("Medical_Keyword_",1:48,sep=""))])

table(All_Data$Number_medical_keywords)

#    0     1      2     3     4     5      6     7     8     9     10     11    12    13    14    16 
#  31247  21430 12573  7046  3652  1793   796   374   132    67    24     5     3     1     1     2

#There seems to be low frequencies in the higher numbers, depending on the model we may want to cap this

All_Data$Number_medical_keywords <- ifelse(All_Data$Number_medical_keywords>7,7,All_Data$Number_medical_keywords)
table(All_Data$Number_medical_keywords)

#    0     1     2     3     4     5      6     7 
#  31247 21430 12573  7046  3652  1793   796   609 



##############################################################
#Step 3: Now that we are finished with feature creation lets recreate train and test
##########################################################

train <- All_Data[All_Data$Train_Flag==1,] #59,381, 131 variables
test <- All_Data[All_Data$Train_Flag==0,] #19,765, 131 variables

set.seed(1234)
train$random <- runif(nrow(train))


##############################################################
#Step 4: Model building - Build a GBM on a random 70% of train and validate on the other 30% of train.
#        This will be an iterative process where you should add/refine/remove features
##########################################################


library(caret)
library("Metrics")

set.seed(998)

inTraining <- createDataPartition(train$Response, p = .07, list = FALSE)
training <- train[ inTraining,]
testing  <- train[-inTraining,]

training <- training[,c("Response","BMI","Wt","Ht","Ins_Age","Number_medical_keywords","Wt_quintile")]

#http://stackoverflow.com/questions/22434850/user-defined-metric-in-caret-package
sqwkSummary <- function (data,
                        lev = NULL,
                        model = NULL) {
  out <- ScoreQuadraticWeightedKappa(as.numeric(data$pred), as.numeric(data$obs))  
  names(out) <- "sqwk"
  flog.debug("sqwk weight - %s ", out)
  out
}



fitControl <- trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 10,
  ## repeated ten times
  repeats = 10,
  summaryFunction = sqwkSummary)



#gbmGrid <-  expand.grid(interaction.depth = c(1, 2, 3, 4),
#                        n.trees = (1:30)*50,
#                        shrinkage = 0.10,
#                        n.minobsinnode = c(5, 10, 15))




set.seed(825)
gbmFit <- train(Response ~ ., data = training,
                 method = "gbm",
                 trControl = fitControl,
                 ## This last option is actually one
                 ## for gbm() that passes through
                 verbose = FALSE,
                 metric = "sqwk")
                #tuneGrid = gbmGrid)




gbmFit

importance <- varImp(gbmFit, scale=FALSE)
# summarize importance
print(importance)
# plot importance
plot(importance)



res <- apply(train, 2, function(col)sum(is.na(col))/length(col))
res[res>0]
length(res[res>0])






