#'################################
#' Random forest for Quora kaggle
#'############################### 


#' Run the xgboost python script 4 for the split. ( the negative class re proportion) THEN
#' export the x_train and y_train as x_train.r.csv and y_train_r.csv respectively



# set your work directory 


# import libraries 
library(data.table)
library(randomForest)

# import data file 
x_train <- fread('x_train_r.csv')
y_train <- fread('y_train_r.csv')
df_test <- fread('test.csv')
df <- cbind(x_train,y_train)


# check for missing data 
numna <- sapply(df, function(x)length(which(is.na(x))))
missingdf <- data.frame(names = colnames(df), missing = numna)
missingdf <- missingdf[order(missingdf$missing, decreasing = TRUE),]
df[is.na(df)] <- 0


# for RF to give classification, observed must be factor 
df$is_duplicate <- as.factor(df$is_duplicate)

# random forest training 
# due to memory constraint, we train in smaller sets
# we set to smaller trees and combine them at the end. total 500 trees 
rf_1 <- randomForest(x = df_2[,-39], y = df_2$is_duplicate, ntree = 50, data = df2)
rf_2 <- randomForest(x = df_2[,-39], y = df_2$is_duplicate, ntree = 50, data = df2)
rf_3 <- randomForest(x = df_2[,-39], y = df_2$is_duplicate, ntree = 50, data = df2)
rf_4 <- randomForest(x = df_2[,-39], y = df_2$is_duplicate, ntree = 50, data = df2)
rf_5 <- randomForest(x = df_2[,-39], y = df_2$is_duplicate, ntree = 50, data = df2)
rf_6 <- randomForest(x = df_2[,-39], y = df_2$is_duplicate, ntree = 50, data = df2)
rf_7 <- randomForest(x = df_2[,-39], y = df_2$is_duplicate, ntree = 50, data = df2)
rf_8 <- randomForest(x = df_2[,-39], y = df_2$is_duplicate, ntree = 50, data = df2)
rf_9 <- randomForest(x = df_2[,-39], y = df_2$is_duplicate, ntree = 50, data = df2)
rf_10 <- randomForest(x = df_2[,-39], y = df_2$is_duplicate, ntree = 50, data = df2)

rf_total <- combine(rf_1,rf_2,rf_3,rf_4,rf_5, rf_6, rf_7, rf_8, rf_9, rf_10) 


# importance 
importance(rf_total)

# import our test and fill the missing data 
x_test <- fread('x_test.csv')
x_test[is.na(x_test)] <- 0

# run the model on the test set 
result <- predict(rf_test, newdata = x_test)

# export the result 
submission <- data.frame(is_duplicate = result, test_id = df_test[,"test_id"])

