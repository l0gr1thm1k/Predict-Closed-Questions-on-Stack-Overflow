import competition_utilities as cu
import features
from sklearn.ensemble import ExtraTreesClassifier
import time

# the values as the should be
#train_file = "train-sample.csv"
#full_train_file = "train.csv"
#test_file = "public_leaderboard.csv"
#submission_file = "basic_benchmark.csv"

classifier_name = "extra_trees"

all_fields_present = 0
update_posteriors = 1
trees_count=50

# the working lab values
# Markdown and title omitted
train_file = "train_no_markdown_no_title_"+str(cu.output_rows_limit)+".csv"
# all fields present
if all_fields_present == 1:
	train_file = "train_all_fields_"+str(cu.output_rows_limit)+".csv"
full_train_file = "train.csv"
test_file = "public_leaderboard.csv"
submission_file = classifier_name + str(trees_count) + "trees" + "_"
if all_fields_present == 1:
	submission_file = classifier_name + str(trees_count) + "trees" + "_all_fields_"

if update_posteriors:
	submission_file = submission_file + "posteriors_updated_";
else:
	submission_file = submission_file + "posteriors_not_updated_";

	
submission_file = submission_file + str(cu.output_rows_limit) + ".csv"
	
feature_names = [ #"BodyLength", # Markdown is required for this feature
                  #"NumTags", # importance: 0.0203914218825
                  "OwnerUndeletedAnswerCountAtPostTime", # importance: 0.189540678306
                  "ReputationAtPostCreation", # importance: 0.27833329466
                  #"TitleLength", # Title is required for this feature
                  "UserAge" # importance: 0.511734605152
                ]

def main():
	start = time.time()
	print("Reading the data from " + train_file)
	data = cu.get_dataframe(train_file)

	print("Extracting features")
	fea = features.extract_features(feature_names, data)

	print("Training the model")
	clf = ExtraTreesClassifier(n_estimators=trees_count, max_features=len(feature_names), max_depth=None, min_samples_split=1, compute_importances=True, bootstrap=False, random_state=0, n_jobs=-1, verbose=2)
	clf.fit(fea, data["OpenStatus"])

	print "Listing feature importances:"
	cu.list_feature_importance(clf,feature_names)
	
	print("Reading test file and making predictions: " + test_file)
	data = cu.get_dataframe(test_file)
	test_features = features.extract_features(feature_names, data)
	probs = clf.predict_proba(test_features)

	if (update_posteriors):
		print("Calculating priors and updating posteriors")
		new_priors = cu.get_priors(full_train_file)
		old_priors = cu.get_priors(train_file)
		probs = cu.cap_and_update_priors(old_priors, probs, new_priors, 0.001)
	
	print("Saving submission to %s" % submission_file)
	cu.write_submission(submission_file, probs)
	
	finish = time.time()
	print "completed in %0.4f seconds" % (finish-start)

if __name__=="__main__":
	main()