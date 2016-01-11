CONTEST INFO

This is the source code of the submission by SkyNet team for the StackExchange contest of closed question prediction.
The code has originally brought us to top 25% (24th/167, 17 entries in team SkyNet), when the contest page had listed all registered teams.
Current situation: we score 24-th/46 (16 entries in team SkyNet)

Details of the contest:

http://www.kaggle.com/c/predict-closed-questions-on-stack-overflow

LIBRARY

We have used scikit, pandas and numpy of Python, as well as other auxiliary helper libraries: csv, datetime, math, re and subprocess.

MODELS TRAINED

1. The winning algorithm was the logistic regresssion based classifier, see logit_classifier.py. 
2. Other classifiers tried didn't score any better, but are still supplied for reference and history. 
The tried classifier algorithms: 
   Gradient Boosting
   MNB
   KNN
   Perceptron (Neural Network)
   MultiTaskElasticNet
   ExtraTreesClassifier 

EVALUATION

The metric used on the contest was Multiclass Logarithmic Loss (https://www.kaggle.com/wiki/MultiClassLogLoss)
We implemented the metric in eval.py.

We have also ran the k-fold evaluation in order to get a sense of how the algorithm performs. This way we could do as many runs a day
as we wanted without wasting the daily submissions (two a day). Usually our quality prediction was a little more optimistic than that of the
actual submission, but served as a good compass in the feature selection as well as tuning the classifiers process.

OTHER IDEAS (LINGUISTIC FEATURES)

In the described above classifiers we have been trying mostly the numeric features. Since a questions is written in natural language, another
idea was to craft textual linguistic features. One such feature is a question's text perplexity calculated against the accepted and closed (rejected)
questions. The text perplexity can be calc'd using SRILM (see language_modelling directory). The pre-trained models of various sort are included.

Some further steps might have included topical classification via extracting keywords and phrases, but wasn't tried. Feel free to try it.

