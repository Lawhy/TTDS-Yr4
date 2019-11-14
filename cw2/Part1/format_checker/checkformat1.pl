$usage = "perl checkformat1.pl file\n";

$file = $ARGV[0] or die $usage;

open(IN,$file) or die "Cannot fine specified file\n";
$line = <IN>;
if($line !~ /^\tP\@10\tR\@50\tr\-Precision\tAP\tnDCG\@10\tnDCG\@20/i){
	die "Error in header line\n";
}
$n=1;
while(<IN>){
	if($n<11){
		if(!/$n(\t[0-9\.]+){6}$/){
			die "Error in line of query $n";
		}
	}
	elsif($n==11){
		if(!/mean(\t[0-9\.]+){6}$/i){
			die "Error in the line of the mean value";
		}
	}
	$n++;
}

print "File format is OK\n";