from sklearn.linear_model import LogisticRegression
import competition_utilities as cu
import pandas as pd
import numpy as np
import features
import time
import sys
from sklearn.cross_validation import KFold
import eval

feature_names = [ "BodyLength",
                  #"ppl",
                "NumTags",
                "OwnerUndeletedAnswerCountAtPostTime",
                "ReputationAtPostCreation",
                #"TitleLength",
                "UserAge",
                #"QuickAndDirtyTag",
                #"CodeSnippetsTag",
                #"BusinessTag"
                ]

                
# order of elements is very important for the output
ques_status = ['not a real question','not constructive','off topic','open','too localized']

# old stuff
#train_file = "train_no_markdown_no_title_"+str(cu.output_rows_limit)+".csv"
full_train_file = "train.csv"
#test_file = "public_leaderboard.csv"

do_cross_validation = 1
all_fields_present = 1
is_full_train_set  = 0

# Markdown and title omitted
train_file = "train_no_markdown_no_title_"+str(cu.output_rows_limit)+".csv"
# all fields present
if all_fields_present == 1:
    train_file = "train_all_fields_"+str(cu.output_rows_limit)+".csv"
#full_train_file = "train_no_markdown_no_title_all.csv"
test_file = "public_leaderboard.csv"
submission_file = "logit_public_leaderboard_"+str(cu.output_rows_limit)+".csv"
if all_fields_present == 1:
    submission_file = "logit_public_all_fields_"+str(cu.output_rows_limit)+".csv"

use_low_mem = 1
_chunksize = 10000
    
if is_full_train_set == 1:
    train_file = full_train_file
    submission_file = "logistic_regression_full_train_body_length.csv"

def main():
    start = time.time()

    if (use_low_mem == 1):
        data_iter = cu.iter_data_frames(train_file, _chunksize)
    
        i = _chunksize
        fea = None
        y = []        
        
        for train_data in data_iter:
            print "About to have processed: " + str(i)
            print("Extracting features")
            if fea is None:
                fea = features.extract_features(feature_names, train_data)
            else:
                fea = fea.append(features.extract_features(feature_names, train_data))
            for element in train_data['OpenStatus']:
                for index, status in enumerate(ques_status):
                    if element == status: y.append(index)
        
            i = i + _chunksize
    else:
        print "Reading train data and its features from: " + train_file
        data = cu.get_dataframe(train_file)
        fea = features.extract_features(feature_names,data)
        print "Collecting statuses"
        y = []
        for element in data["OpenStatus"]:
                for index, status in enumerate(ques_status):
                    if element == status:
                        y.append(index)

    if do_cross_validation == 1:
        logit = LogisticRegression(penalty='l2', dual=False, C=1.0, class_weight=None,
                                       fit_intercept=True, intercept_scaling=1, tol=0.0001)

        print 'starting 10 fold verification'
        # Dividing the dataset into k = 10 folds for cross validation
        kf = KFold(len(y),k = 10)
        fold = 0
        result_sum = 0
        for train_index,test_index in kf:
            fold += 1
            X_train = []
            X_test = []
            y_train = []
            y_test = []
            for i in train_index:
                temp = []
                for feature_name in feature_names:
                    if feature_name == 'BodyLength':
                        temp.append(fea['BodyMarkdown'][i])
                    elif feature_name == 'TitleLength':
                        temp.append(fea['Title'][i])
                    else:
                        temp.append(fea[feature_name][i])
                X_train.append(temp)
                y_train.append(y[i])
                
            for i in test_index:
                temp = []
                for feature_name in feature_names:
                    if feature_name == 'BodyLength':
                        temp.append(fea['BodyMarkdown'][i])
                    elif feature_name == 'TitleLength':
                        temp.append(fea['Title'][i])
                    else:
                        temp.append(fea[feature_name][i])
                X_test.append(temp)
                y_test.append(y[i])
            
            print "fitting this fold's data"
            
            rf.fit(X_train, y_train)
            y_test = vectorize_actual(y_test)
            
            #_pred_probs = denormalize(rf.predict_proba(X_test))
            _pred_probs = rf.predict_proba(X_test)
            
            print("Calculating priors and updating posteriors")
            #new_priors = cu.get_priors(full_train_file)
            new_priors = [0.00913477057600471, 0.004645859639795308, 0.005200965546050945, 0.9791913907850639, 0.0018270134530850952]
            old_priors = cu.get_priors(train_file)
            _pred_probs = cu.cap_and_update_priors(old_priors, _pred_probs, new_priors, 0.001)            
            # evaluating the performance
            result = eval.mcllfun(y_test,_pred_probs)
            result_sum += result
            print "MCLL score for fold %d = %0.11f" % (fold,result)
            
        print "Average MCLL score for this classifier = %0.11f" % (result_sum/10)     
    else:
        logit = LogisticRegression(penalty='l2', dual=False, C=1.0, class_weight=None,
                                       fit_intercept=True, intercept_scaling=1, tol=0.0001) # not available: compute_importances=True

        print "Fitting"
        logit.fit(fea, y)
        
        print "Reading test data and features"
        test_data = cu.get_dataframe(test_file)
        test_fea = features.extract_features(feature_names,test_data)

        print "Making predictions"
        global probs
        probs = logit.predict_proba(test_fea)
        
        if is_full_train_set == 0:
            print("Calculating priors and updating posteriors")
            #new_priors = cu.get_priors(full_train_file)
            new_priors = [0.00913477057600471, 0.004645859639795308, 0.005200965546050945, 0.9791913907850639, 0.0018270134530850952]
            old_priors = cu.get_priors(train_file)
            probs = cu.cap_and_update_priors(old_priors, probs, new_priors, 0.001)    

        print "writing submission to " + submission_file
        cu.write_submission(submission_file, probs)

    finish = time.time()
    print "completed in %0.4f seconds" % (finish-start)

def vectorize_actual(y):
    act = []
    for i in range(0,len(y)):
        temp = [0] * 5
        temp[y[i]] = 1
        act.append(temp)
    return act
    
if __name__ =="__main__":
    main()
