
library(futile.logger)
flog.threshold(DEBUG)

rm(list = ls())

library("neuralnet")

library("Metrics")

flog.debug("start loading nn model")
load('model/nnModel_20160210_1206.rda', verbose=T)
flog.debug("finished loading nn model")

load('processed_data/train_nn_data.rda', verbose=T)
load('processed_data/train.rda', verbose=T)
nn_result_data <- data.frame(cbind(
  compute(nnModel,train_nn_data[, -grep("Response", colnames(train_nn_data))])$net.result,
  train$Response))

rm(nnModel)


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
  number = 10,
  ## repeated ten times
  repeats = 10,
  summaryFunction = sqwkSummary)




set.seed(825)
gbmFit <- train(Response ~ ., data = nn_result_data,
                method = "gbm",
                trControl = fitControl,
                verbose = T,
                metric = "sqwk")


load('processed_data/test_nn_data.rda', verbose=T)
load('processed_data/test.rda', verbose=T)


test$Response <-  predict(gbmFit, newdata = compute(nnModel,nn_test_data)$net.result)

submission_file <- test[,c("Id","Response")] #19,765 obs, 2 variables

write.csv(submission_file,"Submission_file.csv",row.names=FALSE)

