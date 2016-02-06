#####################################################################################
### This script is broken into 5 parts and provides example code to:
#   1) Read in the competition data, concatenate the train and test data, define variables as numeric or factor for the gbm
#   2) Create some simple features which can be used in a predictive model
#   3) Recreate train and test now that features have been created on both
#   4) Build a simple GBM on a random 70% of train, validate on the other 30% and calculate the quadratic weighted kappa
#   5) Score the test data and create a submission file
##############################################
###################################################################################



###############################################################################
#Step 1: Read in the data and define variables as either a factor or numeric
#########################################################################################

library(futile.logger)
flog.threshold(DEBUG)

train <- read.csv("data/train.csv",stringsAsFactors = T) #59,381 observations, 128 variables
#test <- read.csv("data/test.csv",stringsAsFactors = T) #19,765 observations, 127 variables - test does not have a response field

train$Train_Flag <- 1 #Add in a flag to identify if observations fall in train data, 1 train, 0 test
#test$Train_Flag <- 0 #Add in a flag to identify if observations fall in train data, 1 train, 0 test
#test$Response <- NA #Add in a column for Response in the test data and initialize to NA


#concatenate train and test together, any features we create will be on both data sets with the same code. This will make scoring easy
#All_Data <- rbind(train,test) #79,146 observations, 129 variables 
All_Data <- train 



#Define variables as either numeric or factor, Data_1 - Numeric Variables, Data_2 - factor variables
Data_1 <- All_Data[,names(All_Data) %in% c("Medical_History_10","Medical_History_2","Product_Info_4",    "Ins_Age",    "Ht",	"Wt",	"BMI",	"Employment_Info_1",	"Employment_Info_4",	"Employment_Info_6",	"Insurance_History_5",	"Family_Hist_2",	"Family_Hist_3",	"Family_Hist_4",	"Family_Hist_5",	"Medical_History_1",	"Medical_History_15",	"Medical_History_24",	"Medical_History_32",paste("Medical_Keyword_",1:48,sep=""))]
Data_2 <- All_Data[,!(names(All_Data) %in% c("Medical_History_10","Medical_History_2","Product_Info_4",	"Ins_Age",	"Ht",	"Wt",	"BMI",	"Employment_Info_1",	"Employment_Info_4",	"Employment_Info_6",	"Insurance_History_5",	"Family_Hist_2",	"Family_Hist_3",	"Family_Hist_4",	"Family_Hist_5",	"Medical_History_1",	"Medical_History_15",	"Medical_History_24",	"Medical_History_32",paste("Medical_Keyword_",1:48,sep="")))]
Data_2<- data.frame(apply(Data_2, 2, as.factor))

All_Data <- cbind(Data_1,Data_2) #79,146 observations, 129 variables

#We don't need Data_1,Data_2,train or test anymore
rm(Data_1,Data_2,train,test)


library("Hmisc")

#convert numeric to factor
All_Data$Medical_History_1 <- addNA(cut2(All_Data$Medical_History_1, m=500,levels.mean=T))
All_Data$Medical_History_2 <- cut2(All_Data$Medical_History_2, m=1500, levels.mean=T)
All_Data$Medical_History_10 <- addNA(cut2(All_Data$Medical_History_10, m=100, levels.mean=T))
All_Data$Medical_History_15 <- addNA(cut2(All_Data$Medical_History_15, m=500, levels.mean=T))
All_Data$Medical_History_24 <- addNA(cut2(All_Data$Medical_History_24, m=500, levels.mean=T))
All_Data$Medical_History_32 <- addNA(cut2(All_Data$Medical_History_32, m=500, levels.mean=T))

All_Data$Employment_Info_1 <- addNA(cut2(All_Data$Employment_Info_1, m=2000, levels.mean=T))
All_Data$Employment_Info_4 <- addNA(cut2(All_Data$Employment_Info_4, m=500, levels.mean=T))
All_Data$Employment_Info_6 <- addNA(cut2(All_Data$Employment_Info_6, m=2000, levels.mean=T))
All_Data$Insurance_History_5  <- addNA(cut2(All_Data$Insurance_History_5, m=1000, levels.mean=T))
All_Data$Family_Hist_2 <- addNA(cut2(All_Data$Family_Hist_2, m=2000, levels.mean=T))
All_Data$Family_Hist_3 <- addNA(cut2(All_Data$Family_Hist_3, m=1000, levels.mean=T))
All_Data$Family_Hist_4 <- addNA(cut2(All_Data$Family_Hist_4, m=2000, levels.mean=T))
All_Data$Family_Hist_5 <- addNA(cut2(All_Data$Family_Hist_5, m=750, levels.mean=T))


##############################################################
#Step 3: Now that we are finished with feature creation lets recreate train and test
##########################################################

train <- All_Data[All_Data$Train_Flag==1,] #59,381, 131 variables
test <- All_Data[All_Data$Train_Flag==0,] #19,765, 131 variables

rm(All_Data)



set.seed(1234)
train$random <- runif(nrow(train))


##############################################################
#Step 4: Model building - Build a GBM on a random 70% of train and validate on the other 30% of train.
#        This will be an iterative process where you should add/refine/remove features
##########################################################


train_70 <- train[train$random <= 0.7,] #41,561 obs
train_30 <- train[train$random > 0.7,] #17,820 obs

rm(train)

#Lets have a look at distribution of response on train_70 and train_30

#round(table(train_70$Response)/nrow(train_70),2)
# 1     2     3      4     5     6    7    8  
#0.10  0.11  0.02  0.02  0.09  0.19  0.13  0.33

#round(table(train_30$Response)/nrow(train_30),2)
#  1     2     3     4     5     6     7    8 
#0.10  0.11  0.02  0.02  0.09  0.19  0.14  0.33


#The response distribtion holds up well across the random split

#Lets build a very simple GBM on train_70 and calculate the performance on train_30


data_columns <- c('Ht','Wt','BMI','Ins_Age'
                  ,paste("Product_Info_", 1:7, sep="")
                  ,paste("Insurance_History_", 1:5, sep=""),paste("Insurance_History_", 7:9, sep="")
                  #                  ,paste("Employment_Info_", 1:6, sep="")
                  #                  ,paste("InsuredInfo_", 1:6, sep="")
                  #                  ,paste("Family_Hist_", 1:5, sep="")
                  #                  ,paste("Medical_History_",1:41,sep="")
                  #                  ,paste("Medical_Keyword_",1:48,sep="")                  
)






f1 <- as.formula(paste0("~0+", paste(data_columns, collapse="+"), "+Response"))

train_70_subset<-subset(train_70,select=append(data_columns, "Response"))
rm(train_70)


nn_train_data=data.frame(model.matrix(f1, train_70_subset, lapply(as.list(Filter(is.factor, train_70_subset)), contrasts,  contrasts = FALSE)))
rm(train_70_subset)


f2 <- as.formula(paste0(paste(names(nn_train_data)[grepl("Response", names(nn_train_data))], collapse="+"), "~", paste(names(nn_train_data)[!grepl("Response", names(nn_train_data))], collapse="+")))


f3 <- as.formula(paste0("~0+", paste(data_columns, collapse="+")))





library("neuralnet")
#debugonce(neuralnet)


library("Metrics")

#nnModel = neuralnet(f2,data=nn_train_data,linear.output=F, lifesign = 'full', threshold=1.5, err.fct='ce')

#save(nnModel, file = format(Sys.time(), "nnModel_%Y%m%d_%I%M.rda"))

nnModel <- load('nnModel_20160206_1013.rda')


library(caret)
nn_result_data <- cbind(compute(nnModel,nn_train_data)$net.result, train_70$Response)

colnames(nn_result_data) <- c(1:8, "Response")


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




set.seed(825)
gbmFit <- train(Response ~ ., data = nn_result_data,
                method = "gbm",
                trControl = fitControl,
                ## This last option is actually one
                ## for gbm() that passes through
                verbose = FALSE,
                metric = "sqwk")
#tuneGrid = gbmGrid)




gbmFit






train_30_subset<-subset(train_30, select=data_columns)
nn_test_data=data.frame(model.matrix(f3, train_30_subset,lapply(as.list(Filter(is.factor, train_30_subset)), contrasts,  contrasts = FALSE)))







train_30$Prediction <- apply(compute(nnModel,nn_test_data)$net.result, 1, which.max)


print(round((table(train_30$Prediction,train_30$Response)/nrow(train_30))*100,1))

#        1    2    3    4    5    6    7    8
#Pred1  1.8  1.0  0.1  0.0  0.5  1.0  0.7  0.5
#Pred2  1.0  1.7  0.0  0.0  0.8  0.4  0.2  0.1
#Pred5  0.8  1.2  0.3  0.0  3.0  1.0  0.1  0.1
#Pred6  2.5  2.9  0.5  0.4  2.2  6.8  4.0  1.7
#Pred7  0.5  0.5  0.1  0.1  0.4  1.2  1.5  0.7
#Pred8  3.9  3.6  0.7  2.0  2.1  8.1  7.2 29.8

#This predicted distribution is different to the actual distribution we looked at above, We have not classed any applications into groups 3 or 4... we would need to improve the model
#Lets calculate the quadratic weighted kappa with this model

library("Metrics")
testKappa <- ScoreQuadraticWeightedKappa(as.numeric(train_30$Prediction),as.numeric(train_30$Response)) #0.3400944

print(testKappa)



