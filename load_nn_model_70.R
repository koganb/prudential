
library(futile.logger)
flog.threshold(DEBUG)

rm(list = ls())




library("neuralnet")

library("Metrics")

flog.debug("start loading nn model")
load('model/nnModel_20160210_1206.rda', verbose=T)
flog.debug("finished loading nn model")

load('processed_data/train_70_nn_data.rda', verbose=T)
load('processed_data/train_70.rda', verbose=T)
load('processed_data/train_30.rda', verbose=T)
load('processed_data/test_30_nn_data.rda', verbose=T)
nn_result_data <- data.frame(cbind(
  compute(nnModel,train_70_nn_data[, -grep("Response", colnames(train_70_nn_data))])$net.result,
  train_70$Response))


library(caret)
colnames(nn_result_data) <- c(1:8, "Response")

nn_result_data$Response <- as.factor(nn_result_data$Response)


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
  number = 3,
  ## repeated one times
  repeats = 1,
  summaryFunction = sqwkSummary)




set.seed(825)
gbmFit <- train(Response ~ ., data = nn_result_data,
                method = "gbm",
                trControl = fitControl,
                verbose = T,
                metric = "sqwk")


train_model <- compute(nnModel,test_30_nn_data)$net.result
colnames(train_model) <- c(1:8)
train_30$Prediction <- predict(gbmFit, newdata = train_model)


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



