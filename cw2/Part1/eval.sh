#!/bin/bash
set -e
if [ $1 == "all" ]; then
    echo "evaluate $1"
    python evaluation.py --relevant_docs_file systems/qrels.txt --system_results systems --system_num all
else
    for i in {1..6}
    do
        python evaluation.py --relevant_docs_file systems/qrels.txt --system_results systems --system_num $i
        perl format_checker/checkformat1.pl S$i.eval
    done
fi
