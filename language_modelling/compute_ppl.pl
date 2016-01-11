#!/cygdrive/c/Perl64/bin/perl

use strict;
use Lib::SRILM qw(count_ppl_and_return_it);

# INPUT
# sentences file
# sentence ids file
# sentences vocab
# language model
print count_ppl_and_return_it($ARGV[0], $ARGV[1], $ARGV[2], $ARGV[3]);
