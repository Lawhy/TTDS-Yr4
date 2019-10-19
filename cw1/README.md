### Instructions

##### Attention! Please first generate index.pkl before running any search code, i.e. run step 1 before step 2 & 3.

##### 1. For the generation of index.pkl and index.txt (Postional Inverted Index):
```
"""Args: --xml: path to the xml file; 
         --st: path to the stop_words list"""
python lexicon.py --xml CW1collection/trec.5000.xml --st CW1collection/englishST.txt
```
##### 2. For the generation of queries.boolean.txt (results of queries via Boolean Search):
```
"""Args: --index: path to the index.pkl file generated in advance
         --st: path to the stop_words list
         --search: search type {bool, rank}
         --query: path to the file containing queries""" 
python search.py --index index.pkl \
                 --st 'CW1collection/englishST.txt' \
                 --search bool \
                 --query 'CW1collection/queries.boolean.txt' 
```
##### 3. For the generation of queries.ranked.txt (results of queires via Ranking based on TF-IDF):
```
"""Args: --index: path to the index.pkl file generated in advance
         --st: path to the stop_words list
         --search: search type {bool, rank}
         --query: path to the file containing queries""" 
python search.py --index index.pkl \
                 --st 'CW1collection/englishST.txt' \
                 --search rank \
                 --query 'CW1collection/queries.ranked.txt'
```
