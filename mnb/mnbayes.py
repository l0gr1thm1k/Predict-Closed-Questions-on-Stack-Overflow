from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
import competition_utilities as cu
import pandas as pd
import numpy as np
import features
import time
import sys
import eval

feature_names = [
					"ReputationAtPostCreation",
					"UserAge",
					"NumTags",
				]

# order of elements is very important for the output
ques_status = ['not a real question','not constructive','off topic','open','too localized']

classifier = "mnbayes"

all_fields_present = 0
is_full_train_set  = 1
do_cross_validation = 1

# Markdown and title omitted
train_file = "train_no_markdown_no_title_"+str(cu.output_rows_limit)+".csv"
# all fields present
if all_fields_present == 1:
	train_file = "train_all_fields_"+str(cu.output_rows_limit)+".csv"
full_train_file = "train_no_markdown_no_title_all.csv"
test_file = "public_leaderboard.csv"
submission_file = classifier + "_public_leaderboard_"+str(cu.output_rows_limit)+".csv"
if all_fields_present == 1:
	submission_file = classifier + "_public_leaderboard_all_fields_"+str(cu.output_rows_limit)+".csv"
	
if is_full_train_set == 1:
	train_file = full_train_file
	submission_file = classifier + "_public_leaderboard_full_train_no_markdown_no_title_reputation.csv"

def main():
	start = time.time()

	print "Reading train data and its features from: " + train_file
	data = cu.get_dataframe(train_file)
	global fea
	fea = features.extract_features(feature_names,data)

	mnbayes = MultinomialNB(alpha=1.0, fit_prior=True)
	
	'''Make sure you have the up to date version of sklearn; v0.12 has the
           predict_proba method; http://scikit-learn.org/0.11/install.html '''   
	
	# Must be array type object. Strings must be converted to
	# to integer values, otherwise fit method raises ValueError
	y = []
	
	for element in data['OpenStatus']:
		for index, status in enumerate(ques_status):
			if element == status: y.append(index)
	
	if do_cross_validation == 1:
		print 'starting 10 fold verification'
		# Dividing the dataset into k = 10 folds for cross validation
		#skf = StratifiedKFold(y,k = 10)
		skf = KFold(len(y),k = 10)
		fold = 0
		result_sum = 0
		for train_index,test_index in skf:
			fold += 1
			X_train = []
			X_test = []
			y_train = []
			y_test = []
			for i in train_index:
				temp = []
				for feature_name in feature_names:
					temp.append(fea[feature_name][i])
				X_train.append(temp)
				y_train.append(y[i])
				
			for i in test_index:
				temp = []
				for feature_name in feature_names:
					temp.append(fea[feature_name][i])
				X_test.append(temp)
				y_test.append(y[i])
			
			mnbayes.fit(X_train, y_train) #, sample_weight=None, class_prior=[0.0091347705760047, 0.0046458596397953, 0.0052009655460509, 0.9791913907850639, 0.0018270134530851])
			y_test = vectorize_actual(y_test)               # vectorize y_test
			
			_pred_probs = mnbayes.predict_proba(X_test)
			# evaluating the performance
			result = eval.mcllfun(y_test,_pred_probs)
			result_sum += result
			print "MCLL score for fold %d = %0.11f" % (fold,result)
			
		print "Average MCLL score for this classifier = %0.11f" % (result_sum/10)
	
		print "Reading test data and features"
		test_data = cu.get_dataframe(test_file)
		test_fea = features.extract_features(feature_names,test_data)
		
		print "Fitting"
		mnbayes.fit(fea,y)#, class_prior=[0.0091347705760047, 0.0046458596397953, 0.0052009655460509, 0.9791913907850639, 0.0018270134530851])
		
		print "Making predictions"
		global probs
		probs = mnbayes.predict_proba(test_fea)

		#if is_full_train_set == 0:
		#	print("Calculating priors and updating posteriors")
		#	new_priors = cu.get_priors(full_train_file)
		#	old_priors = cu.get_priors(train_file)
		#	probs = cu.cap_and_update_priors(old_priors, probs, new_priors, 0.001)

		print "writing submission to " + submission_file
		cu.write_submission(submission_file, probs)
	
	finish = time.time()
	print "completed in %0.4f seconds" % (finish-start)

def fit_data(data, bayes_clf):
	fea = features.extract_features(feature_names,data)

	# Must be array type object. Strings must be converted to
	# to integer values, otherwise fit method raises ValueError
	global y
	y = [] 

	print "Collecting statuses"
	
	for element in data["OpenStatus"]:
			for index, status in enumerate(ques_status):
				if element == status:
					y.append(index)

	print "Fitting"
	bayes_clf.fit(fea, y, sample_weight=None, class_prior=[0.0091347705760047, 0.0046458596397953, 0.0052009655460509, 0.9791913907850639, 0.0018270134530851])
	
	return bayes_clf

'''
	Returns binary representation, where for each sample the OpenStatus's index is set to 1 and the rest values are 0
'''
def binarize(data):
	binarized_data = np.array([])
	for element in data["OpenStatus"]:
		sample_array = np.array([0, 0, 0, 0, 0])
		for index, status in enumerate(ques_status):
			if element == status:
				sample_array[index] = 1
				break
		np.append(binarized_data, sample_array, axis=0)
	return binarized_data

def vectorize_actual(y):
	act = []
	for i in range(0,len(y)):
		temp = [0] * 5
		temp[y[i]] = 1
		act.append(temp)
	return act

if __name__ =="__main__":
	main()
