import competition_utilities as cu
import csv
import datetime
from dateutil import parser
import features
import numpy as np
import pandas as pd
import re
import math
import subprocess

def camel_to_underscores(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

##############################################################
###### FEATURE FUNCTIONS
##############################################################

def body_length(data):
    #try:
        return pd.DataFrame.from_dict({"BodyLength" : data["BodyMarkdown"].apply(str).apply(len)})
    #except TypeError:
    #    return 0

def num_tags(data):
    return pd.DataFrame.from_dict({"NumTags": [sum(map(lambda x:
                    pd.isnull(x), row)) for row in (data[["Tag%d" % d
                    for d in range(1,5)]].values)] } ) ["NumTags"]

def title_length(data):
    #try:
        return pd.DataFrame.from_dict({"TitleLength" : data["Title"].apply(len)})
    #except TypeError:
    #    return 0
    
def user_age(data):
    return pd.DataFrame.from_dict({"UserAge": (data["PostCreationDate"].apply(parser.parse) - data["OwnerCreationDate"].apply(parser.parse)).apply(lambda x: x.total_seconds()).apply(math.fabs)})

'''
def quick_and_dirty_tag(data):
    for row in (data[["Tag%d" % d for d in range(1,6)]].values):
        if row is not None:
                if row.any() == "quick-and-dirty":
                    return pd.DataFrame([dict(QuickAndDirtyTag=1)])
    return pd.DataFrame([dict(QuickAndDirtyTag=0)])
def code_snippets_tag(data):
    for row in (data[["Tag%d" % d for d in range(1,6)]].values):
        if row.any() == "code-snippets":
            return pd.DataFrame([dict(CodeSnippetsTag=1)])
    return pd.DataFrame([dict(CodeSnippetsTag=0)])
def business_tag(data):
    ret_features = pd.DataFrame(index=data.index)
    for row in (data[["Tag%d" % d for d in range(1,6)]].values):
        if row.any() == "business":
            ret_features = ret_features.append(pd.DataFrame("1"))
        else:
            ret_features = ret_features.append(pd.DataFrame("0"))
    return ret_features
    
def quick_and_dirty_tag(data):
    return pd.DataFrame.from_dict({"QuickAndDirtyTag": [lambda x:
                    int(x == "quick-and-dirty") for x in (data[["Tag%d" % d
                    for d in range(1,6)]].values)] } ) ["QuickAndDirtyTag"]

def ppl(data):
    try:
        return data["BodyMarkdown"].apply(get_ppl)
    except TypeError:
        return 1000000.0
        
def get_ppl(text):
    ppl = 1000000.0
    # prepare the data
    x = cu.remove_chars(text)    
    y = cu.tokenize_punc(x)
    
    language_models = [
    "not_a_real_questionLMBin",
    "too_localizedLMBin",
    "not_constructiveLMBin",
    "off_topicLMBin",
    "openLMBin"
    ]
    
    ppls = []
    
    f = open("single_q.txt", "w")
    f.write(y.lstrip().upper())
    f.close()
    
    #create_single_q_vocab("single_q.txt", "single_q_vocab.txt")
    
    for lm in language_models:
        p = subprocess.Popen('perl language_modelling/compute_ppl.pl single_q.txt single_q_vocab.txt language_modelling/lm/' + lm, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in p.stdout.readlines():
            ppl = float(line)
            ppls.append(ppl)
        retval = p.wait()
    print ppls
    ppl = min(ppls)
    print "min = " + str(ppl)
    return ppl

def create_single_q_vocab(text, text_vocab):
    p = subprocess.Popen('ngram-count ' + ' -text ' + text + ' -write-vocab ' + text_vocab)
###########################################################
'''

def extract_features(feature_names, data):
    fea = pd.DataFrame(index=data.index)
    for name in feature_names:
        if name in data:
            #fea = fea.join(data[name])
            fea = fea.join(data[name].apply(math.fabs))
        else:
            #try:
                fea = fea.join(getattr(features, camel_to_underscores(name))(data))
            #except TypeError:
            #    pass
    return fea

if __name__=="__main__":
    feature_names = [ "BodyLength"
                    , "NumTags"
                    , "OwnerUndeletedAnswerCountAtPostTime"
                    , "ReputationAtPostCreation"
                    , "TitleLength"
                    , "UserAge"
                    ]
              
    data = cu.get_dataframe("private_leaderboard_massaged.csv") #cu.get_dataframe("train-sample_October_9_2012_v2_massaged.csv")
    features = extract_features(feature_names, data)
    print(features)
    #print features['UserAge']
    #print features['BodyLength']
    #print features['TitleLength']
    print features['NumTags']

