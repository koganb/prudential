
library(futile.logger)
flog.threshold(DEBUG)

rm(list = ls())


train <- read.csv("data/train.csv",stringsAsFactors = T) #59,381 observations, 128 variables
test <- read.csv("data/test.csv",stringsAsFactors = T) #19,765 observations, 127 variables - test does not have a response field

train$Train_Flag <- 1 #Add in a flag to identify if observations fall in train data, 1 train, 0 test
test$Train_Flag <- 0 #Add in a flag to identify if observations fall in train data, 1 train, 0 test
test$Response <- NA #Add in a column for Response in the test data and initialize to NA


#concatenate train and test together, any features we create will be on both data sets with the same code. This will make scoring easy
All_Data <- rbind(train,test) #79,146 observations, 129 variables 


#Define variables as either numeric or factor, Data_1 - Numeric Variables, Data_2 - factor variables
Data_1 <- All_Data[,names(All_Data) %in% c("Medical_History_10","Medical_History_2","Product_Info_4",    "Ins_Age",    "Ht",	"Wt",	"BMI",	"Employment_Info_1",	"Employment_Info_4",	"Employment_Info_6",	"Insurance_History_5",	"Family_Hist_2",	"Family_Hist_3",	"Family_Hist_4",	"Family_Hist_5",	"Medical_History_1",	"Medical_History_15",	"Medical_History_24",	"Medical_History_32",paste("Medical_Keyword_",1:48,sep=""))]
Data_2 <- All_Data[,!(names(All_Data) %in% c("Medical_History_10","Medical_History_2","Product_Info_4",	"Ins_Age",	"Ht",	"Wt",	"BMI",	"Employment_Info_1",	"Employment_Info_4",	"Employment_Info_6",	"Insurance_History_5",	"Family_Hist_2",	"Family_Hist_3",	"Family_Hist_4",	"Family_Hist_5",	"Medical_History_1",	"Medical_History_15",	"Medical_History_24",	"Medical_History_32",paste("Medical_Keyword_",1:48,sep="")))]
Data_2<- data.frame(apply(Data_2, 2, as.factor))

All_Data <- cbind(Data_1,Data_2) #79,146 observations, 129 variables

#We don't need Data_1,Data_2,train or test anymore
rm(Data_1,Data_2,train,test)


library("Hmisc")

#convert numeric to factor
All_Data$Medical_History_1 <- addNA(cut2(All_Data$Medical_History_1, m=7000,levels.mean=T))
All_Data$Medical_History_2 <- cut2(All_Data$Medical_History_2, m=7000, levels.mean=T)
All_Data$Medical_History_10 <- addNA(cut2(All_Data$Medical_History_10, m=100, levels.mean=T))
All_Data$Medical_History_15 <- addNA(cut2(All_Data$Medical_History_15, m=1500, levels.mean=T))
All_Data$Medical_History_24 <- addNA(cut2(All_Data$Medical_History_24, m=500, levels.mean=T))
All_Data$Medical_History_32 <- addNA(cut2(All_Data$Medical_History_32, m=500, levels.mean=T))

All_Data$Employment_Info_1 <- addNA(cut2(All_Data$Employment_Info_1, m=8000, levels.mean=T))
All_Data$Employment_Info_4 <- addNA(cut2(All_Data$Employment_Info_4, m=1500, levels.mean=T))
All_Data$Employment_Info_6 <- addNA(cut2(All_Data$Employment_Info_6, m=7000, levels.mean=T))
All_Data$Insurance_History_5  <- addNA(cut2(All_Data$Insurance_History_5, m=5000, levels.mean=T))
All_Data$Family_Hist_2 <- addNA(cut2(All_Data$Family_Hist_2, m=4500, levels.mean=T))
All_Data$Family_Hist_3 <- addNA(cut2(All_Data$Family_Hist_3, m=4500, levels.mean=T))
All_Data$Family_Hist_4 <- addNA(cut2(All_Data$Family_Hist_4, m=5000, levels.mean=T))
All_Data$Family_Hist_5 <- addNA(cut2(All_Data$Family_Hist_5, m=2500, levels.mean=T))

test <- All_Data[All_Data$Train_Flag==0,] #19,765, 131 variables

set.seed(1234)

train <- All_Data[All_Data$Train_Flag==1,] #59,381, 131 variables
train$random <- runif(nrow(train))
train_70 <- train[train$random <= 0.7,] #41,561 obs
train_30 <- train[train$random > 0.7,] #17,820 obs




data_columns <- c('Ht','Wt','BMI','Ins_Age'
                  ,paste("Product_Info_", 1:7, sep="")
                  ,paste("Insurance_History_", 1:5, sep=""),paste("Insurance_History_", 7:9, sep="")
                  ,paste("Employment_Info_", 1:6, sep="")
                  ,paste("InsuredInfo_", 1:6, sep="")
                  ,paste("Family_Hist_", 1:5, sep="")
                  ,paste("Medical_History_",1:41,sep="")
                  ,paste("Medical_Keyword_",1:48,sep="")                  
)


train_formula <- as.formula(paste0("~0+", paste(data_columns, collapse="+"), "+Response"))


flog.info("start saving nn_train70")
train_70<-subset(train_70,select=append(data_columns, "Response"))
save(train_70, file = "train_70.rda")
train_70_nn_data=data.frame(model.matrix(train_formula, train_70, lapply(as.list(Filter(is.factor, train_70)), contrasts,  contrasts = FALSE)))
rm(train_70)
save(train_70_nn_data, file = "train_70_nn_data.rda")
rm(train_70_nn_data)
flog.info("end saving nn_train70")


flog.info("start saving nn_train")
train<-subset(train,select=append(data_columns, "Response"))
save(train, file = "train.rda")
train_nn_data=data.frame(model.matrix(train_formula, train, lapply(as.list(Filter(is.factor, train)), contrasts,  contrasts = FALSE)))
rm(train)
save(train_nn_data, file = "train_nn_data.rda")
rm(train_nn_data)
flog.info("end saving nn_train")


test_formula <- as.formula(paste0("~0+", paste(data_columns, collapse="+")))

flog.info("start saving test30")
test_30<-subset(train_30, select=data_columns)
save(train_30, file = "train_30.rda")
rm(train_30)
save(test_30, file = "test_30.rda")
test_30_nn_data=data.frame(model.matrix(test_formula, test_30,lapply(as.list(Filter(is.factor, test_30)), contrasts,  contrasts = FALSE)))
rm(test_30)
save(test_30_nn_data, file = "test_30_nn_data.rda")
rm(test_30_nn_data)
flog.info("end saving test30")

flog.info("start saving test")
test<-subset(test, select=data_columns)
save(test, file = "test.rda")
test_nn_data=data.frame(model.matrix(test_formula, test,lapply(as.list(Filter(is.factor, test)), contrasts,  contrasts = FALSE)))
rm(test)
save(test_nn_data, file = "test_nn_data.rda")
rm(test_nn_data)
flog.info("end saving test")


