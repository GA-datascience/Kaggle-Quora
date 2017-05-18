#'################################
#' Random forest for Quora kaggle
#'############################### 
# http://stackoverflow.com/questions/33881053/how-to-get-randomforest-model-output-in-probability-using-caret

#' Run the xgboost python script 4 for the split. ( the negative class re proportion) THEN
#' export the x_train and y_train as x_train.r.csv and y_train_r.csv respectively



# set your work directory 


# import libraries 
library(data.table)
library(randomForest)

# import data file 
x_train <- fread('x_train_r.csv')
y_train <- fread('y_train_r.csv')
df_test <- fread('test_corrected.csv')
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
df$ab_norm_wmd[which(is.infinite(df$ab_norm_wmd))] <- 0.9999999999
df$ab_wmd[which(is.infinite(df$ab_wmd))] <- 10

set.seed(100)
rf_1 <- randomForest(x = df[,-56], y = df$is_duplicate, ntree = 50, data = df)
set.seed(200)
rf_2 <- randomForest(x = df[,-56], y = df$is_duplicate, ntree = 50, data = df)
set.seed(300)
rf_3 <- randomForest(x = df[,-56], y = df$is_duplicate, ntree = 50, data = df)
set.seed(400)
rf_4 <- randomForest(x = df[,-56], y = df$is_duplicate, ntree = 50, data = df)
set.seed(500)
rf_5 <- randomForest(x = df[,-56], y = df$is_duplicate, ntree = 50, data = df)

rf_total <- combine(rf_1,rf_2,rf_3,rf_4,rf_5)
remove(rf_1,rf_2,rf_3,rf_4,rf_5)

set.seed(123)
rf_6 <- randomForest(x = df[,-56], y = df$is_duplicate, ntree = 50, data = df)
rf_total <- combine(rf_total,rf_6)
remove(rf_6)

set.seed(246)
rf_7 <- randomForest(x = df[,-56], y = df$is_duplicate, ntree = 50, data = df)
rf_total <- combine(rf_total,rf_7)
remove(rf_7)

set.seed(234)
rf_8 <- randomForest(x = df[,-56], y = df$is_duplicate, ntree = 50, data = df)
rf_total <- combine(rf_total,rf_8)
remove(rf_8)

set.seed(456)
rf_9 <- randomForest(x = df[,-56], y = df$is_duplicate, ntree = 50, data = df)
rf_total <- combine(rf_total,rf_9)
remove(rf_9)

set.seed(619)
rf_10 <- randomForest(x = df[,-56], y = df$is_duplicate, ntree = 50, data = df)
rf_total <- combine(rf_total,rf_10)
remove(rf_10)

set.seed(123123)
rf_11 <- randomForest(x = df[,-56], y = df$is_duplicate, ntree = 50, data = df)
rf_total <- combine(rf_total,rf_11)
remove(rf_11)


rf_total <- combine(rf_1,rf_2,rf_3,rf_4,rf_5, rf_6, rf_7, rf_8, rf_9, rf_10) 


# importance 
importance(rf_total)

# import our test and fill the missing data 
x_test <- fread('x_test.csv')

x_test$ab_norm_wmd[which(is.infinite(x_test$ab_norm_wmd))] <- 0.9999999999
x_test$ab_wmd[which(is.infinite(x_test$ab_wmd))] <- 10

x_test[is.na(x_test)] <- 0

# run the model on the test set 
result <- predict(rf_total, newdata = x_test, type = "prob")

# export the result 
submission <- data.table(is_duplicate = result[,2],  df_test[,"test_id"])
#write.csv(submission,'randomforest.csv',row.names = FALSE) 
fwrite(submission,'11randomforests.csv')
