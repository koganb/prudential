rm(list = ls())  #remove environment

load('processed_data/train_nn_data.rda',verbose = T)

f2 <- as.formula(paste0(paste(names(train_nn_data)[grepl("Response", names(train_nn_data))], collapse="+"), "~", paste(names(train_nn_data)[!grepl("Response", names(train_nn_data))], collapse="+")))


library("neuralnet")
#debugonce(neuralnet)


library("Metrics")


nnModel = neuralnet(f2,data=train_nn_data,linear.output=F, lifesign = 'full', stepmax=500000, rep=1, hidden=c(1), threshold=10, err.fct='ce')

save(nnModel, file = format(Sys.time(), "model/nnModel_All_%Y%m%d_%I%M.rda"))




