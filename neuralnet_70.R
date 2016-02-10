rm(list = ls())  #remove environment

load('processed_data/train_70_nn_data.rda',verbose = T)

f2 <- as.formula(paste0(paste(names(train_70_nn_data)[grepl("Response", names(train_70_nn_data))], collapse="+"), "~", paste(names(train_70_nn_data)[!grepl("Response", names(train_70_nn_data))], collapse="+")))


library("neuralnet")
#debugonce(neuralnet)


library("Metrics")


nnModel = neuralnet(f2,data=train_70_nn_data,linear.output=F, lifesign = 'full', stepmax=500000, rep=1, hidden=c(10), threshold=0.3, err.fct='ce')

save(nnModel, file = format(Sys.time(), "model/nnModel_70_%Y%m%d_%I%M.rda"))


rm(train_70_nn_data)

load('processed_data/test_30_nn_data.rda', verbose=T)
load('processed_data/train_30.rda', verbose=T)

train_30$Prediction <- apply(compute(nnModel,test_30_nn_data)$net.result, 1, which.max)


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



