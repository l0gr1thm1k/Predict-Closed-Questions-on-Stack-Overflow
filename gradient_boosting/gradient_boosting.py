import competition_utilities as cu
import features
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import KFold
import eval
import winsound

train_file = "train-sample_October_9_2012_v2.csv"
full_train_file = "train.csv"
test_file = "private_leaderboard.csv"
submission_file = "gradient_boosting_score_improved_by_14_points_private_leaderboard_full_train.csv"

do_cross_validation = 0
do_full_train = 1
use_low_mem = 1
_chunksize = 10000

if do_full_train == 1:
    train_file = full_train_file

feature_names = [ "BodyLength",
                "NumTags",
                "OwnerUndeletedAnswerCountAtPostTime",
                "ReputationAtPostCreation",
                "UserAge",
                ]

ques_status = ['not a real question','not constructive','off topic','open','too localized']

def main():

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
        print("Reading the data from:" + train_file)
        data = cu.get_dataframe(train_file)
        print("Extracting features")
        fea = features.extract_features(feature_names, data)
        y = []
        for element in data['OpenStatus']:
            for index, status in enumerate(ques_status):
                if element == status: y.append(index)    

    if do_cross_validation == 1:
        depth = len(feature_names)
        print "depth=" + str(depth)
        rf = GradientBoostingClassifier(loss='deviance', learn_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=1, min_samples_leaf=1, max_depth=depth, init=None, random_state=None)

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
            
            print "Fitting for fold " + str(fold)
            
            rf.fit(X_train, y_train)
            y_test = vectorize_actual(y_test)
            
            _pred_probs = rf.predict_proba(X_test)
            print("Calculating priors and updating posteriors")
            #new_priors = cu.get_priors(full_train_file)
            # priors distribution over classes based on the training set
            #new_priors = [0.00913477057600471, 0.004645859639795308, 0.005200965546050945, 0.9791913907850639, 0.0018270134530850952]
            # priors distribution over classes based on the updated training set's last month
            new_priors = [0.03410911204982466, 0.01173872976800856, 0.018430671606251586, 0.926642216133641, 0.009079270442274271]
            old_priors = cu.get_priors(train_file)
            _pred_probs = cu.cap_and_update_priors(old_priors, _pred_probs, new_priors, 0.001)            
            # evaluating the performance
            result = eval.mcllfun(y_test,_pred_probs)
            result_sum += result
            print "MCLL score for fold %d = %0.11f" % (fold,result)
            
        print "depth=" + str(depth)
        print "Average MCLL score for this classifier = %0.11f" % (result_sum/10)    
    else:
        #rf = RandomForestClassifier(n_estimators=50, verbose=0, compute_importances=True, n_jobs=-1)
        rf = GradientBoostingClassifier(loss='deviance', learn_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=1, min_samples_leaf=1, max_depth=len(feature_names), init=None, random_state=None)
        
        rf.fit(fea,y)
        
        print("Reading test file " + test_file + " and making predictions")
        data = cu.get_dataframe(test_file)
        test_features = features.extract_features(feature_names, data)
        probs = rf.predict_proba(test_features)

        # commented out, because we want to adjust probabilities to the last month data anyway
        #if do_full_train == 0:
        print("Calculating priors and updating posteriors")
        new_priors = cu.get_priors(full_train_file)
        old_priors = cu.get_priors(train_file)
        probs = cu.cap_and_update_priors(old_priors, probs, new_priors, 0.001)
    
        print("Saving submission to %s" % submission_file)
        cu.write_submission(submission_file, probs)
        

def vectorize_actual(y):
    act = []
    for i in range(0,len(y)):
        temp = [0] * 5
        temp[y[i]] = 1
        act.append(temp)
    return act

def sos():
    for i in range(0, 3): 
        winsound.Beep(2000, 1000) 
        for i in range(0, 3): 
            winsound.Beep(2000, 400) 
            for i in range(0, 3): winsound.Beep(2000, 100)
            
if __name__=="__main__":
    main()