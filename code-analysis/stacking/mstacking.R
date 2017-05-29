###########################
##########################
# manual stacking 
#########################

#http://amsantac.co/blog/en/2016/10/22/model-stacking-classification-r.html

library(data.table)
library(caret)
library(caretEnsemble)
library(extraTrees)
library(randomForest)
library(xgboost)

library(Matrix)

set.seed(12357)
# load in the train file 
setwd("D:/python/kaggle competitions/quora")
train <- fread('train_corrected.csv')


# did a 0.7 split in python. Now we import them 
# note that i call the 0.7 training and 0.3 testing set 
trainingdf <- fread('x_train_train_r.csv')
ylabel_train <- fread('y_train_train_r.csv') 

trainingdf <- cbind(trainingdf, ylabel_train)

# for the 'test' set 
testingdf <- fread('x_train_test_r.csv')
ylabel_test <- fread('y_train_test_r.csv')

testingdf <- cbind(testingdf, ylabel_test)

# for xg boost 
trainingdf_xg <- trainingdf
testingdf_xg <- testingdf
# change both response to factor 
trainingdf$label <- as.factor(trainingdf$label)
testingdf$label <- as.factor(testingdf$label)




# random forest=====================================================================  
rf_1 <- randomForest(x = trainingdf[,-67], y = trainingdf$label, ntree = 50, data = df)
rf_2 <- randomForest(x = trainingdf[,-67], y = trainingdf$label, ntree = 50, data = df)
rf_3 <- randomForest(x = trainingdf[,-67], y = trainingdf$label, ntree = 50, data = df)
rf_4 <- randomForest(x = trainingdf[,-67], y = trainingdf$label, ntree = 50, data = df)
rf_5 <- randomForest(x = trainingdf[,-67], y = trainingdf$label, ntree = 50, data = df)
rf_6 <- randomForest(x = trainingdf[,-67], y = trainingdf$label, ntree = 50, data = df)
rf_7 <- randomForest(x = trainingdf[,-67], y = trainingdf$label, ntree = 50, data = df)
rf_8 <- randomForest(x = trainingdf[,-67], y = trainingdf$label, ntree = 50, data = df)
rf_9 <- randomForest(x = trainingdf[,-67], y = trainingdf$label, ntree = 50, data = df)
rf_10 <- randomForest(x = trainingdf[,-67], y = trainingdf$label, ntree = 50, data = df)

rf_total <- combine(rf_1,rf_2,rf_3,rf_4,rf_5, rf_6, rf_7, rf_8, rf_9, rf_10) 



rf_prediction <- predict(rf_total, testingdf)


#xgboost=========================================================================== 
train <- as.matrix(trainingdf_xg)
test <- as.matrix(testingdf_xg)
train <- as(train, "sparseMatrix")
test <- as(test,"sparseMatrix")

xgb_train <- xgb.DMatrix(data = train[,1:66], label = train[,"label"])
xgb_test <- xgb.DMatrix(data = test[,1:66])

param <- list(
  seed = 12357,
  objective="binary:logistic",
  booster="gbtree",
  eta=0.1, 
  max.depth=5, 
  eval_metric = "logloss"
)


watchlist <- list(train = xgb_train)

bst <- xgb.train(params = param, data = xgb_train, nrounds = 1000, watchlist =watchlist, verbose = 1, print_every_n = 50)

# fit on the 'test' set for prediction 

xgb_prediction <- predict(bst,xgb_test)


#############################
## Correlation betw models 
############################


results <- resample(list(rf = rf_total, xgb = bst))
modelCor(results)




###########################
## Level 1 modelling 
###########################

#' Making us of the prediction on the 0.3 of the training set. Now we 
#' will make us of the prediction as well as test label as features 
#' 
source('logloss.R') 
control <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions= "final", classProbs=TRUE, summaryFunction = LogLosSummary)
predict_features <- data.frame(rf_prediction, xgb_prediction, testingdf$label)
# run a level 1 model on glm using predict_features 
greedy_stack <- train(label ~. ,method = "glm", metric = "LogLoss", trControl = control, data= predict_features)




x_test <- fread("x_test.csv")
stacked_prediction <- predict(greedy_stack,x_test)