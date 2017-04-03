# features selection 
# http://machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/


# AFTER you ran your: 
    #x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size = 0.2, random_state = random)
    # or step 4. Before xgboosting 



from numpy import loadtxt
from numpy import sort
from xgboost import XGBClassifier
from sklearn.cross_validation import train_test_split 
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel






# fit model all training data
model = XGBClassifier()
model.fit(x_train, y_train)


# make predictions for test data and evaluate 
y_pred = model.predict(x_valid)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_valid, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# fit model using each importance as a threhold 
thresholds = sort(model.feature_importances_)
for thresh in thresholds:
	# select features using threshold
	selection = SelectFromModel(model, threshold=thresh, prefit=True)
	select_X_train = selection.transform(x_train.fillna(0))
    # train model
	selection_model = XGBClassifier()
	selection_model.fit(select_X_train, y_train)
    # eval model
	select_X_test = selection.transform(x_valid.fillna(0))
	y_pred = selection_model.predict(select_X_test)
	predictions = [round(value) for value in y_pred]
	accuracy = accuracy_score(y_valid, predictions)
	print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))