'''
KNN classifier. Takes the non-conventional way of completely removing the text 
BodyMarkdown field and concentrating on other features.
'''
import competition_utilities as cu
import features, time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import StratifiedKFold
import eval_tool   #Dmitry's eval tool

feature_names = ['ReputationAtPostCreation',
		 'UserAge',
		 'TitleLength',
		 'BodyLength',
		 'OwnerUndeletedAnswerCountAtPostTime']

def main():
	start = time.time()

	result_sum = 0
	
	data = cu.get_dataframe("data/train-sample.csv")
	#test_data = cu.get_dataframe("data/public_leaderboard.csv")   #use this for evaluating public_leaderboard
	
	print 'data loaded'
	
	fea = features.extract_features(feature_names, data)
	#test_fea = features.extract_features(feature_names,test_data)  #use this for evaluating public_leaderboard
	
	print 'features extracted'
	
	knn = KNeighborsClassifier(n_neighbors=10,weights='distance')
	
	# Must be array type object. Strings must be converted to
	# to integer values, otherwise fit method raises ValueError
	y = []
	ques_status = ['open', 'too localized', 'not constructive','off topic','not a real question']
	for element in data['OpenStatus']:
		for index, status in enumerate(ques_status):
			if element == status: y.append(index)
	
	print 'starting 10 fold verification'
	# Dividing the dataset into k = 10 folds for cross validation
	skf = StratifiedKFold(y,k = 10)
	fold = 0
	for train_index, test_index in skf:
		fold += 1
		X_train = []
		X_test = []
		y_train = []
		y_test = []
		for i in train_index:
			temp = []
			temp.append(fea['ReputationAtPostCreation'][i])
			temp.append(fea['UserAge'][i])
			temp.append(fea['Title'][i])
			temp.append(fea['BodyMarkdown'][i])
			temp.append(fea['OwnerUndeletedAnswerCountAtPostTime'][i])
			X_train.append(temp)
			y_train.append(y[i])
			
		for i in test_index:
			temp = []
			temp.append(fea['ReputationAtPostCreation'][i])
			temp.append(fea['UserAge'][i])
			temp.append(fea['Title'][i])
			temp.append(fea['BodyMarkdown'][i])
			temp.append(fea['OwnerUndeletedAnswerCountAtPostTime'][i])
			X_test.append(temp)
			y_test.append(y[i])
		
		y_test = vectorize_actual(y_test)               # vectorize y_test
		knn.fit(X_train,y_train)                        # train the classifier
		predictions = knn.predict_proba(X_test)         # predict the test fold
		
		# evaluating the performance
		result = eval_tool.mcllfun(y_test,predictions)
		result_sum += result
		print "MCLL score for fold %d = %0.11f" % (fold,result)
		
	print "Average MCLL score for this classifier = %0.11f" % (result_sum/10)
	finish = time.time()
	print "completed in %0.4f seconds" % (finish-start)
	
	### Use this code for evaluting public_leaderboard	
	'''knn.fit(fea,y)
	print 'classifier trained. now predicting'
	predictions = knn.predict_proba(test_fea)
	cu.write_submission("knn_v0.3_public_leader_1.csv",predictions)'''
	

def vectorize_actual(y):
	act = []
	for i in range(0,len(y)):
		temp = [0] * 5
		temp[y[i]] = 1
		act.append(temp)
	return act

if __name__ == "__main__":
	main()
