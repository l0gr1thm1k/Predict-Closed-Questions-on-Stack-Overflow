from sklearn.linear_model import Perceptron
import competition_utilities as cu
import pandas as pd
import numpy as np
import features
import time
import sys

feature_names = [
					"ReputationAtPostCreation",
					"OwnerUndeletedAnswerCountAtPostTime",
					"UserAge",
				]

# order of elements is very important for the output
ques_status = ['not a real question','not constructive','off topic','open','too localized']
#ques_status = ['off topic']

classifier = "perceptron"

all_fields_present = 0
is_full_train_set  = 0

# Markdown and title omitted
train_file = "train_no_markdown_no_title_"+str(cu.output_rows_limit)+".csv"
# all fields present
if all_fields_present == 1:
	train_file = "train_all_fields_"+str(cu.output_rows_limit)+".csv"
full_train_file = "train.csv"
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

	percep = Perceptron(penalty=None, alpha=0.0001, fit_intercept=False, n_iter=5, shuffle=False, verbose=1, eta0=1.0, n_jobs=-1, seed=0, class_weight="auto", warm_start=False)

	X = []
	for i in data["OwnerUndeletedAnswerCountAtPostTime"]:
		X.append([i])
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
	percep.fit(fea, y)
	
	'''Make sure you have the up to date version of sklearn; v0.12 has the
           predict_proba method; http://scikit-learn.org/0.11/install.html '''   
	
	print "Reading test data and features"
	test_data = cu.get_dataframe(test_file)
	test_fea = features.extract_features(feature_names,test_data)

	print "Making predictions"
	global probs
	#probs = percep.predict_proba(test_fea) # only available for binary classification
	probs = percep.predict(test_fea)
	# shape of probs is [n_samples]
	# convert probs to shape [n_samples,n_classes]
	probs = np.resize(probs, (len(probs) / 5, 5))
	
	#if is_full_train_set == 0:
	#	print("Calculating priors and updating posteriors")
	#	new_priors = cu.get_priors(full_train_file)
	#	old_priors = cu.get_priors(train_file)
	#	probs = cu.cap_and_update_priors(old_priors, probs, new_priors, 0.001)	

	print "writing submission to " + submission_file
	cu.write_submission(submission_file, probs)
	finish = time.time()
	print "completed in %0.4f seconds" % (finish-start)
	
if __name__ =="__main__":
	main()
