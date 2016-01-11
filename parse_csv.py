''' competition_utilities class is shared by Stack Exchange on github, 
    the link for which can be found on contest home/data page '''
import os
import competition_utilities as cu
import pandas as pd
import csv
import time

main_path = "."
_chunk_size = 10000
#skip_fields = ['BodyMarkdown', 'Title']
skip_fields = ['PostId', 'PostCreationDate', 'OwnerUserId', 'OwnerCreationDate', 'OwnerUndeletedAnswerCountAtPostTime', 'Title', 'BodyMarkdown', 'Tag1', 'Tag2', 'Tag3', 'Tag4', 'Tag5', 'PostClosedDate']
question_status = ['open', 'too localized', 'not constructive', 'off topic',
			   'not a real question']
output_all_entries = 1
			   
train_file = "train.csv"
output_sampled_file = "train_all_fields_"+str(cu.output_rows_limit)+".csv" #"train_no_markdown_no_title_"+str(cu.output_rows_limit)+".csv"
if output_all_entries == 1:
	output_sampled_file = "train_no_markdown_only_reputation_all.csv"


''' this function returns an iterable parser for csv file. Each iteration
	gives us a DataFrame object for that particular chunk '''
def iter_questions(file_name,ques_status):
	df_iter = pd.io.parsers.read_csv(os.path.join(main_path, file_name), iterator=True, chunksize=_chunk_size)
	return (question[1] for df in df_iter for question in df[df["OpenStatus"] == ques_status].iterrows())
	
''' sample function to read the BodyMarkDown field of the questions that are "too localized" '''
def sample():
	for q in iter_questions("split2/test2.csv","open"):
		print q['BodyMarkdown']

		
def sample_by_class(writer,class_name,read_num=-1):
	i = 0
	print "reading class:" + class_name
	for q in iter_questions("data/" + filename_in,class_name):
		if (i % _chunk_size == 0):
			print str(i)
		if i == read_num:
			break
		values = []
		for field in q.keys():
			if (field not in skip_fields):
				values.append(q[field])
			else:
				values.append("''")
		writer.writerow(values)
		i = i + 1
	print "written out total for this class: " + str(i)
	return i
		
if __name__=="__main__":

	start = time.time()
	
	filename_in = train_file
	filename_out = os.path.join(main_path, "data", output_sampled_file)
	
	writer = csv.writer(open(filename_out, "w"), lineterminator="\n")
	writer.writerow(cu.get_header(filename_in))
	
	total_written = 0
	if output_all_entries == 0:
		per_class_limit = cu.output_rows_limit / len(question_status)
	else:
		per_class_limit = -1
	for status in question_status:
		total_written = total_written + sample_by_class(writer,status,per_class_limit)

	print "total rows written:" + str(total_written)
	finish = time.time()
	print "completed in %0.4f seconds" % (finish-start)

	