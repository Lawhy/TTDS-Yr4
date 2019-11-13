$usage = "perl checkformat2.pl All.eval\n";

$file = $ARGV[0] or die $usage;

open(IN,$file) or die "Cannot fine specified All.eval\n";
$line = <IN>;
if($line !~ /^\tP\@10\tR\@50\tr\-Precision\tAP\tnDCG\@10\tnDCG\@20/i){
	die "Error in header line\n";
}
$n=1;
while(<IN>){
	if($n<7){
		if(!/S$n(\t[0-9\.]+){6}$/i){
			die "Error in line of S$n";
		}
	}
	$n++;
}

print "File format is OK\n";