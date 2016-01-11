from __future__ import division
from collections import Counter
import csv
import dateutil
import numpy as np
import os
import pandas as pd
import re

base = "d:\\Projects\\other_projects\\machine_learning\\stackexchange_closed_q"

data_path = base + "\\data"
submissions_path = base + "\\submission"

if not data_path or not submissions_path:
    raise Exception("Set the data and submission paths in competition_utilities.py!")

global output_rows_limit
output_rows_limit = 100000

def remove_chars(sentence):
    """
    Remove unwanted characters; not necessary, but makes data appear more
    uniform; chars_to_remove can always be added to if anything pops up.
    """
    chars_to_remove = ['\r', '\n', '.']
    for x in chars_to_remove:
        if x in sentence:
            sentence = sentence.replace(x, ' ') + " ."
    return re.sub('\s+', ' ', sentence)

def tokenize_punc(sentence):
    """
    Split punctuation into separate tokens; One difficulty in constructing
    models with the StackOverflow data is that occasionally these occurances
    of puncuation are a piece of code and should maybe not be tokenized. 
    
    Conversely, the use of the n-gram option "-prune-lowprobs" to eliminate 
    many hapax legomenon may make this concern a non-issue 
    """
    punc = ['.', ',', '!', '?', '--', '(', ')', '{', '}']
    for x in punc:
        if x in sentence:
            sentence = sentence.replace(x, ' ' + x + ' ')
    return sentence

def get_sentences(input_string):
    """
    Split string into sentences via RegEx; Convert text to uppercase, this
    greatly constrains the langauge model to a smaller number of N-grams
    """
    pattern = r"([\s\S]*?[\.\?!]\s+)"
    sentences = re.findall(pattern, input_string.upper())
    return sentences

def parse_date_maybe_null(date):
    if date:
        return dateutil.parser.parse(date)
    return None
'''
def parse_body_markdown(contents):
	print "Skipping BodyMarkdown";
	return None
'''
df_converters = {#"PostCreationDate": dateutil.parser.parse,
                 #"OwnerCreationDate": dateutil.parser.parse,
                 #"PostClosedDate": parse_date_maybe_null, # this raises ValueError("unknown string format")
				 # "BodyMarkdown": parse_body_markdown
				 }

def get_reader(file_name="train-sample.csv"):
    reader = csv.reader(open(os.path.join(data_path, file_name)))
    header = reader.next()
    return reader

def get_header(file_name="train-sample.csv"):
    reader = csv.reader(open(os.path.join(data_path, file_name)))
    header = reader.next()
    return header

def get_closed_count(file_name):
    return sum(1 for q in iter_closed_questions(file_name))

def iter_closed_questions(file_name):
    df_iter = pd.io.parsers.read_csv(os.path.join(data_path, file_name), iterator=True, chunksize=1000)
    return (question[1] for df in df_iter for question in df[df["OpenStatus"] != "open"].iterrows())

def iter_open_questions(file_name):
    df_iter = pd.io.parsers.read_csv(os.path.join(data_path, file_name), iterator=True, chunksize=1000)
    return (question[1] for df in df_iter for question in df[df["OpenStatus"] == "open"].iterrows())

def get_dataframe(file_name="train-sample.csv"):
    return pd.io.parsers.read_csv(os.path.join(data_path, file_name), converters = df_converters)

def iter_data_frames(file_name, _chunksize):
    return pd.io.parsers.read_csv(os.path.join(data_path, file_name), iterator=True, chunksize=_chunksize, converters = df_converters)

def get_priors(file_name):
    closed_reasons = [r[3] for r in get_reader(file_name)]
    closed_reason_counts = Counter(closed_reasons)
    reasons = sorted(closed_reason_counts.keys())
    total = len(closed_reasons)
    priors = [closed_reason_counts[reason]/total for reason in reasons]
    return priors

def write_sample(file_name, header, sample):
    writer = csv.writer(open(os.path.join(data_path, file_name), "w"), lineterminator="\n")
    writer.writerow(header)
    writer.writerows(sample)

def update_prior(old_prior,  old_posterior, new_prior):
    evidence_ratio = (old_prior*(1-old_posterior)) / (old_posterior*(1-old_prior))
    new_posterior = new_prior / (new_prior + (1-new_prior)*evidence_ratio)
    return new_posterior

def cap_and_update_priors(old_priors, old_posteriors, new_priors, epsilon):
    old_posteriors = cap_predictions(old_posteriors, epsilon)
    old_priors = np.kron(np.ones((np.size(old_posteriors, 0), 1)), old_priors)
    new_priors = np.kron(np.ones((np.size(old_posteriors, 0), 1)), new_priors)
    evidence_ratio = (old_priors*(1-old_posteriors)) / (old_posteriors*(1-old_priors))
    new_posteriors = new_priors / (new_priors + (1-new_priors)*evidence_ratio)
    new_posteriors = cap_predictions(new_posteriors, epsilon)
    return new_posteriors

def cap_predictions(probs, epsilon):
    probs[probs > 1-epsilon] = 1-epsilon
    probs[probs < epsilon] = epsilon
    row_sums = probs.sum(axis=1)
    probs = probs / row_sums[:, np.newaxis]
    return probs

def write_submission(file_name, predictions):
    writer = csv.writer(open(os.path.join(submissions_path, file_name), "w"), lineterminator="\n")
    writer.writerows(predictions)

def list_feature_importance(classifier,feature_names):
	try:
		for i in range(0,len(feature_names)):
			print feature_names[i] + "'s importance is " + str(classifier.feature_importances_[i])
	except AttributeError:
		print "feature_importances_ property isn't found in this classifier"
