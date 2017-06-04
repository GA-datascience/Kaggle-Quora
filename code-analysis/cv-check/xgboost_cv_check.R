# cv for quora xgboost 

library(caret)
source("logloss.R")


set.seed(12357)

fitcontrol <- trainControl(method = "cv", number - 5, savePredictions = "final",classProbs = TRUE, summaryFunction = LogLosSummary)
xgb_grid <- expand.grid(nrounds= 1000, 
                        max_depth= 5 ,eta= 0.1, 
                        gamma= 0, colsample_bytree = c(1), 
                        min_child_weight = c(1), subsample = 1)

model_xgb <- train(is_duplicate~., data=trainset ,method="xgbTree", trControl= fitcontrol, tuneGrid= xgb_grid, metric="LogLoss",maximize = FALSE)
