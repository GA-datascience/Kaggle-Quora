# cross validation for kaggle 
# parameter tuning 

# https://www.analyticsvidhya.com/blog/2016/01/xgboost-algorithm-easy-steps/
# https://www.kaggle.com/khozzy/rossmann-store-sales/xgboost-parameter-tuning-template/run/90168/notebook




setwd("D:/python/kaggle competitions/quora")
require(xgboost)
require(data.table)



# note that this files are exported after the split steps in python 
x_train <- fread('xtrain_r.csv')
y_train <- fread('label.csv')

# change to matrix

#require(Matrix)
#x_train_matrix <- sparse.model.matrix(data=x_train)

# converting to dmatrix object for modelling later
d_train <- xgb.DMatrix(data = data.matrix(x_train), label = y_train$is_duplicate)



# define param 
param <- list(
  objective="binary:logistic",
  booster="gbtree",
  eta=0.15, # Control the learning rate
  max.depth=5, # Maximum depth of the tree
  subsample=0.8 # subsample ratio of the training instance
)


# training 




history <- xgb.cv(
  data=d_train,
  params = param,
  early_stop_round=30, # training with a validation set will stop if the performance keeps getting worse consecutively for k rounds
  nthread=4, # number of CPU threads
  nround=500, # number of trees
  print_every_n = 10,
  verbose=1, #  show partial info
  nfold=5, # number of CV folds
  metrics = "logloss", # custom evaluation metric
  maximize=FALSE # the lower the evaluation score the better
)

# https://www.kaggle.com/nigelcarpenter/allstate-claims-severity/farons-xgb-starter-ported-to-r/discussion


best_nrounds = which.min(history$evaluation_log$train_logloss_mean) 

