import pickle, re, os, sys

class LM():
    def __init__(self):
        # dictionary of OpenStatus keys and BodyMarkdown values 
        self.markdown = pickle.load(open("BodyMarkdown.pickle", "rb"))
        self.preprocess()

    def preprocess(self):
        """
        Use the markdown dictionary as the source text; This command could be
        modified to take a file or Python obj argument as the source text.
        
        Make pre processed data files with cleaned up data; Text is formatted to 
        uppercase, extraneous newline characters are removed and puncuation is 
        tokenized; The resulting text files are used to construct language models
        """
        for key in self.markdown.keys():
            # data goes to this file   
            f = open(key + ".txt", "wb")
            # clean the data up before writing to file
            largeString = "\n".join(self.markdown[key])
            sentences = self.get_sentences(largeString)
            for sentence in sentences:
                x = self.remove_chars(sentence)    
                y = self.tokenize_punc(x)
                # write data to file sentence by sentence
                f.write(y.lstrip() + '\n')
            f.close()

    def get_sentences(self, input_string):
        """
        Split string into sentences via RegEx; Convert text to uppercase, this
        greatly constrains the langauge model to a smaller number of N-grams
        """
        pattern = r"([\s\S]*?[\.\?!]\s+)"
        sentences = re.findall(pattern, input_string.upper())
        return sentences


    def remove_chars(self, sentence):
        """
        Remove unwanted characters; not necessary, but makes data appear more
        uniform; chars_to_remove can always be added to if anything pops up.
        """
        chars_to_remove = ['\r', '\n']
        for x in chars_to_remove:
            if x in sentence:
                sentence = sentence.replace(x, ' ')
        return sentence

    def tokenize_punc(self, sentence):
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


        
    def make_model(self, ques_status):
        """
        Make a language model based on question OpenStatus with SRILM's ngram-count
        command; Resulting model is readable by the ngram command.
        """
        command = '/cygdrive/d/Projects/machine_translation/Lab/argmax_langmodel/ngram-count.exe -text "%s.txt" -lm "%sLM" -kndiscount1 -kndiscount2 -kndiscount3' % (ques_status, ques_status)
        os.system(command)    

if __name__ == "__main__":
    import time
    start = time.time()
    myModel = LM()
    for key in myModel.markdown.keys():
        myModel.make_model(key)
    finish = time.time()
    print "Finished generating models in %0.2f sec" % (finish-start)
