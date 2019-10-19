### Instructions

For the generation of the index.pkl and index.txt (Postional Inverted Index):
```
# Args: --xml path to the xml file; --st path to the stop_words list
python lexicon.py --xml CW1collection/trec.5000.xml --st CW1collection/englishST.txt
```

python search.py --index index.pkl --search bool --query 'CW1collection/queries.boolean.txt' 

python search.py --index index.pkl --search rank --query 'CW1collection/queries.ranked.txt'
