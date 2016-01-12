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


train <- read.csv("data/train.csv",stringsAsFactors = T) #59,381 observations, 128 variables
test <- read.csv("data/test.csv",stringsAsFactors = T) #19,765 observations, 127 variables - test does not have a response field

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

#Function to sum across rows for variables defined
psum <- function(...,na.rm=FALSE) { 
    rowSums(do.call(cbind,list(...)),na.rm=na.rm) }
All_Data$Number_medical_keywords <- psum(All_Data[,c(paste("Medical_Keyword_",1:48,sep=""))])




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


train_70 <- train[train$random <= 0.7,] #41,561 obs
train_30 <- train[train$random > 0.7,] #17,820 obs

#Lets have a look at distribution of response on train_70 and train_30

round(table(train_70$Response)/nrow(train_70),2)
# 1     2     3      4     5     6    7    8  
#0.10  0.11  0.02  0.02  0.09  0.19  0.13  0.33

round(table(train_30$Response)/nrow(train_30),2)
#  1     2     3     4     5     6     7    8 
#0.10  0.11  0.02  0.02  0.09  0.19  0.14  0.33


#The response distribtion holds up well across the random split

#Lets build a very simple GBM on train_70 and calculate the performance on train_30




columns <- columns <- c('Ht','Wt','BMI','Ins_Age', "Family_Hist_1","Number_medical_keywords" , "Employment_Info_2","Employment_Info_3", "Employment_Info_5", paste("Medical_History_",33:41,sep=""), paste("Medical_History_",2:9,sep=""), paste("Medical_History_",11:14,sep=""),  paste("Medical_History_",16:23,sep=""), paste("Medical_History_",25:31,sep=""), paste("Insurance_History_",1:4,sep=""), paste("Insurance_History_",7:9,sep=""),paste("InsuredInfo_",1:7,sep=""), "Product_Info_1", "Product_Info_3", "Product_Info_5", "Product_Info_6","Product_Info_7")

ideal <- class.ind(train_70$Response)
nnModel = nnet(subset(train_70,select=columns), ideal, size=13, softmax=TRUE, maxit=1000)


train_30$Prediction <-predict(nnModel, subset(train_30,select=columns), type="class")


round((table(train_30$Prediction,train_30$Response)/nrow(train_30))*100,1)

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



##############################################################
#Step 5: Score test data
#        Repeat step 4 till we are happy with the model. It could be beneficial to rerun the final model on all of the training data and use cross validation etc.
#        Once we are happy, score the testing data and create a submission file
##########################################################


Prediction_Object <- predict(GBM_train,test,GBM_train$opt_tree,type="response")

#an array with probability of falling into each class for each observation
#We want to classify each application, a trivial approach would be to take the class with the highest predicted probability for each application

test$Response <- apply(Prediction_Object, 1, which.max)

round(table(test$Response)/nrow(test),2)
# 1      2     5     6      7    8 
#0.06  0.04   0.06  0.21  0.05  0.57


submission_file <- test[,c("Id","Response")] #19,765 obs, 2 variables

write.csv(submission_file,"Submission_file.csv",row.names=FALSE)
