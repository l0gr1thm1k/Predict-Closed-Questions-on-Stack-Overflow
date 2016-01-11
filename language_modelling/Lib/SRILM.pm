package Lib::SRILM;

use strict;

use base 'Exporter';
our @EXPORT = ('load_ppl', 'count_ppl', 'count_ppl_and_return_it');

# 0 or 1
my $verbose = 0;
 
#my $srilm_base = "/cygdrive/d/Projects/LanguageModellingLab/bin/cygwin";
my $srilm_base = "/cygdrive/d/Projects/machine_translation/Lab/argmax_langmodel";

sub load_ppl
{
	my $ppl_file = shift;
	my %sent_to_ppl;
	
	open PPLFILE, "<".$ppl_file or die $!;
	my($str);
	while(defined($str = <PPLFILE>) && !($str =~ /^file.+/)) {
		chomp $str;
		
		next if ($str eq "");
		
		# first line is sentence
		my $sent = $str;
		my %params;
		# second line is of form:
		# 1 sentences, 6 words, 0 OOVs
		# we skip it
		$str = <PPLFILE>;
		# third line is of form:
		# 0 zeroprobs, logprob= -12.727 ppl= 65.7868 ppl1= 132.179
		# we extract parameters from it
		$str = <PPLFILE>;
		print $str if ($verbose);
		$str =~ /logprob= (.[\d\.]+)/;
		print $1."\n" if ($verbose);
		$params{'logprob'} = $1;
		
		$str =~ /ppl= (.[\d\.]+)/;
		print $1."\n" if($verbose);
		$params{'ppl'} = $1;
		
		$str =~ /ppl1= (.[\d\.]+)/;
		$params{'ppl1'} = $1;
		print $1."\n" if($verbose);
		$sent_to_ppl{$sent} = \%params;
	}
	
	close PPLFILE;
	
	return \%sent_to_ppl;
}

sub count_ppl
{
	my $sentences_file = shift;
	my $lang_model     = shift;
	my $output_file    = shift;
	
	my $cmd = "$srilm_base/ngram -lm $lang_model -ppl $sentences_file -debug 1 > $output_file";
	print "running $cmd\n" if ($verbose);
	print system($cmd);
}

sub count_ppl_and_return_it
{
	my $sentences_file    = shift;
	my $sentence_ids_file = shift;
	my $sentences_vocab   = shift;
	my $lang_model        = shift;
	
	my $ppl = 100000;
	
	return $ppl if (! -f $sentences_file || ! -f $lang_model || ! -f $sentence_ids_file);
	
	my $sent_ids_arr = &load_sent_ids($sentence_ids_file);
	
	#my $cmd = "ngram -lm $lang_model -ppl $sentences_file -prune-lowprobs -debug 0";
	my $cmd = "ngram -lm $lang_model -ppl $sentences_file -order 1 -debug 1";
	print $cmd . "\n" if ($verbose);
	my $output = qx($cmd);
	#my $output = `$cmd`;
	
	print $output if ($verbose);
	
	#die;
	
	## Now you can loop through each line and
	## parse the $line variable to extract the info you are looking for.
	
	my $sent_order_number = 0;
	
	foreach my $str (split /[\n]+/, $output) {
		## Regular expression magic to grab what you want
		chomp $str;
		
		next if ($str =~ /^file.+/);

		next if ($str eq "");

		# second line is of form:
		# 1 sentences, 6 words, 0 OOVs
		# we skip it
		#$str = <PPLFILE>;
		next if ($str =~ /.+OOVs$/);
		# third line is of form:
		# 0 zeroprobs, logprob= -12.727 ppl= 65.7868 ppl1= 132.179
		# we extract parameters from it
		#$str = <PPLFILE>;
		print $str if ($verbose);
		$str =~ /logprob= (.[\d\.]+)/;
		print $1."\n" if ($verbose);
		#$params{'logprob'} = $1;

		$str =~ /ppl= (.[\d\.]+)/;
		print $1."\n" if($verbose);
		#$params{'ppl'} = $1;
		$ppl = $1;
		
		print $sent_ids_arr->[$sent_order_number] . ", " . $ppl . "\n";

		$str =~ /ppl1= (.[\d\.]+)/;
		#$params{'ppl1'} = $1;
		print $1."\n" if($verbose);
		$sent_order_number++;
	}
	print "Sentences processed: " . $sent_order_number . "\n";
	return $ppl;
}

sub load_sent_ids {
	my $ids_file = shift;
	my @ret_arr;
	open IDS, "<" . $ids_file or die $!;
	while(defined(my $line = <IDS>)) {
		chomp $line;
		push @ret_arr, $line;
	}
	close IDS;
	return \@ret_arr;
}

1;