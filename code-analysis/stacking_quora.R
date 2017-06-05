# stacking for quora 



# loading library and data 

library(caret)
library(caretEnsemble)
library(data.table)
library(Metrics)
source('logloss.R')



# going parallel
library(parallel)
detectCores() # number of max cores we have 
library(doParallel)

#Sys.setenv(JAVA_HOME='C:\\Program Files\\Java\\jre1.8.0_131')
#library(rJava)

trainset <- fread('x_train_r.csv')
label <- fread('y_train_r.csv')
testset <- fread('x_test.csv')

trainset$is_duplicate <- label$is_duplicate
trainset$is_duplicate <- ifelse(trainset$is_duplicate == 1, "Y", "N")
trainset$is_duplicate <- as.factor(trainset$is_duplicate)


# check for missing/ infinite / Nan 

trainset[is.na(trainset)] <- 0 
trainset$ab_wmd[which(is.infinite(trainset$ab_wmd))] <- 0
trainset$ab_norm_wmd[which(is.infinite(trainset$ab_norm_wmd))] <- 0

testset[is.na(testset)] <- 0 
testset$ab_wmd[which(is.infinite(testset$ab_wmd))] <- 0
testset$ab_norm_wmd[which(is.infinite(testset$ab_norm_wmd))] <- 0


missingnum <- sapply(testset, function(x)length(which(is.na(x))))
infinitenum <- sapply(testset, function(x)length(which(is.infinite(x))))
nannum <- sapply(testset, function(x)length(which(is.nan(x))))
check <- data.frame(name = colnames(testset), missing = missingnum, infinite = infinitenum, nan = nannum)

# Modelling======================================



fitcontrol <- trainControl(method = "cv", number =5, savePredictions = "final",
                           classProbs = TRUE, verboseIter = TRUE, summaryFunction = LogLosSummary)

# glm logloss from cv = 0.2340028 public: 0.2==========================================
set.seed(12357)
model_lr <- train(is_duplicate ~. , data = trainset, 
                  method = "glm", 
                  trControl = fitcontrol,
                  metric = "LogLoss", 
                  maximize = FALSE )



prediction_lr <- predict(model_lr, testset, type = "prob")

# write features out 

#meta features

meta_features <- data.frame(is_duplicate = trainset$is_duplicate, lr = model_lr$pred$Y[order(model_lr$pred$rowIndex)])

meta_test <- data.frame(lr = prediction_lr$Y )

# test write out 
submission <- fread('sample_submission.csv')
submission$is_duplicate <- prediction_lr$Y
fwrite(submission, "submission_lr.csv")



# Ridge Regression # c(0.01, 0.1,1,10, 100)==============================================
lambda <- 10^seq(3, -2 , by = -0.1)

# cv lambda 0.01 = 0.2521,
ridge_grid <- expand.grid(alpha = 0, lambda = lambda  )
set.seed(12357)
model_ridge <- train(is_duplicate~. , data = trainset, method = "glmnet", 
                     metric = "LogLoss", tuneGrid = ridge_grid, trControl = fitcontrol, maximize = FALSE )

prediction_ridge <- predict(model_ridge, testset, type = "prob")
# write features out 
meta_features$ridge <- model_ridge$pred$Y[order(model_ridge$pred$rowIndex)]

meta_test$ridge <- prediction_ridge$Y

submission$is_duplicate <- prediction_ridge$Y
fwrite(submission, "submission_ridge")


# extra trees
extra_grid <- expand.grid()
set.seed(12357)
model_etrees <- train(is_duplicate ~., data = trainset, method = "extraTrees",
                      metric = "LogLoss", trControl = fitcontrol, maximize = FALSE )



# GBM=======================================================================================

# Max shrinkage for gbm
nl = nrow(trainset)
max(0.01, 0.1*min(1, nl/10000)) # 0.1 for shrinkage
# Max Value for interaction.depth c(1,3,6,9,10)
floor(sqrt(NCOL(trainset))) 
set.seed(12357)

gbm.grid <-  expand.grid(interaction.depth = c(1, 3, 6, 8),
                         n.trees = (0:50)*50, 
                         shrinkage = seq(.0005, .1, .0005),
                         n.minobsinnode = 10) # you can also put something like c(5, 10, 15, 20)

# The final values used for the model were n.trees = 150, interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 10.
model_gbm <- train(is_duplicate ~. , data = trainset, method = "gbm", trControl = fitcontrol, metric = "LogLoss", maximize = FALSE)



prediction_gbm <- predict(model_gbm, testset, type = "prob" )
submission$is_duplicate <- prediction_gbm$Y
fwrite(submission, "submission_gbm")


meta_features$gbm <- model_gbm$pred$Y[order(model_gbm$pred$rowIndex)]

meta_test$gbm <- prediction_gbm$Y

# knn ====================================================================================================



set.seed(12357)
cl <- makeCluster(detectCores() - 1)

registerDoParallel(cl)
model_knn <- train(is_duplicate~., data=trainset ,method="knn", trControl= fitcontrol, 
                   tuneLength = 5, metric="LogLoss",maximize = FALSE)


stopCluster(cl)


















# xgb=============================================================================================

set.seed(12357)




xgb_grid <- expand.grid(nrounds= 1000, 
                        max_depth= 5 ,eta= 0.1, 
                        gamma= 0, colsample_bytree = c(1), 
                        min_child_weight = c(1), subsample = 1)

model_xgb <- train(is_duplicate~., data=trainset ,method="xgbTree", trControl= fitcontrol, 
                   tuneGrid= xgb_grid, metric="LogLoss",maximize = FALSE, verbose = TRUE)


prediction_xgb <- predict(model_xgb, testset, type = "prob")

# extract features

meta_features$xgb <- model_xgb$pred$Y[order(model_xgb$pred$rowIndex)]

meta_test$xgb <- prediction_xgb$Y


# stack==========================================================================================================
meta_features <- fread('meta_features_1.csv')
meta_test <- fread('meta_test_1.csv'
                   )
layer1_model <- train(is_duplicate ~  xgb + gbm + ridge2, data = meta_features, method = "glm", metric = "LogLoss", trControl= fitcontrol,
                      maximize = FALSE, tuneLength = 3 )

prediction_layer1 <- predict(layer1_model, meta_test, type = "prob")
submission$is_duplicate <- prediction_layer1$Y
fwrite(submission, "submission_stacked.csv")

