### Instructions to use the IR Evaluation Code (Please run it in the Python3 environment).


```
"""
args:
1. --relevant_docs_file: the path to qrels.txt  # e.g. systems/qrels.txt
2. --system_results: teh path to all system results # e.g. systems (which contains S1-6.results
3. --system_num: if set 'all', it generates All.eval, else if set a number (1-6), it generates S#Num.eval.
"""
# Assume S1-6.results and qrels.txt are in the directory systems.
# To generate All.eval
python evaluation.py --relevant_docs_file systems/qrels.txt --system_results systems --system_num all
# Output:
Loading system results: systems/S1.results...
Loading system results: systems/S2.results...
Loading system results: systems/S3.results...
Loading system results: systems/S4.results...
Loading system results: systems/S5.results...
Loading system results: systems/S6.results...
# And a local file All.eval


# To generate S3.eval
python evaluation.py --relevant_docs_file systems/qrels.txt --system_results systems --system_num 3
# Output:
Loading system results: systems/S3.results...
       P@10   R@50 r-Precision     AP nDCG@10 nDCG@20
1     0.300  0.667       0.500  0.518   0.660   0.733
2     0.600  1.000       0.625  0.750   0.832   0.897
3     0.000  1.000       0.000  0.056   0.000   0.240
4     0.800  0.875       0.700  0.690   0.684   0.704
5     0.300  0.429       0.143  0.104   0.233   0.233
6     0.400  1.000       0.417  0.465   0.132   0.449
7     0.000  0.000       0.000  0.000   0.000   0.000
8     0.800  1.000       1.000  1.000   0.780   0.780
9     0.800  0.900       0.900  0.756   0.464   0.584
10    0.100  0.800       0.200  0.174   0.417   0.488
mean  0.410  0.767       0.448  0.451   0.420   0.511
# And a local file S3.eval
```
