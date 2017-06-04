# cv for quora xgboost 

library(caret)
source("logloss.R")


set.seed(12357)

fitcontrol <- trainControl(method = "cv", number =5, savePredictions = "final",
                           classProbs = TRUE, verboseIter = TRUE, summaryFunction = LogLosSummary)



xgb_grid <- expand.grid(nrounds= 1000, 
                        max_depth= 5 ,eta= 0.1, 
                        gamma= 0, colsample_bytree = c(1), 
                        min_child_weight = c(1), subsample = 1)

model_xgb <- train(is_duplicate~., data=trainset ,method="xgbTree", trControl= fitcontrol, 
                   tuneGrid= xgb_grid, metric="LogLoss",maximize = FALSE, verbose = TRUE)



# So 5 fold cv =      is equivalent to public LB = 0.151 
# Let us benchmark this 
# Accuracy   Kappa      LogLoss  
#  0.9321653  0.7620084  0.1655568

#Tuning parameter 'nrounds' was held constant at a value of 1000
#Tuning parameter 'max_depth' was held constant at a
# a value of 0
#Tuning parameter 'colsample_bytree' was held constant at a value of 1
#Tuning parameter
# 'min_child_weight' was held constant at a value of 1
#Tuning parameter 'subsample' was held constant at a value of 1
