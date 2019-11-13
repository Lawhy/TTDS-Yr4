import pandas as pd
import numpy as np
import re
from collections import defaultdict, OrderedDict
from decimal import *
from argparse import ArgumentParser


class Eval:
    
    def __init__(self, relevant_docs_file):
        self.relevant_docs = self.read_relevant_docs(relevant_docs_file)
        self.current_system_results = None
    
    @staticmethod
    def parse_relevant_doc(line):
        """parse each line of qrels.txt into usable information"""
        query_num = re.findall(r'([0-9]+?):', line)
        relevant_docs_n_scores = re.findall(r'\(([0-9]+?),([0-9]+?)\)', line)
        return query_num[0], relevant_docs_n_scores
    
    @classmethod
    def read_relevant_docs(cls, relevant_docs_file):
        """read relevant documents and the associated relevance scores for all queries"""
        with open(relevant_docs_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        results = dict()
        for line in lines:
            query_num, relevant_docs_n_scores = cls.parse_relevant_doc(line)
            results[int(query_num)] = [(int(item[0]), int(item[1])) for item in relevant_docs_n_scores]
        return results
    
    @staticmethod
    def parse_system_result(line):
        """parse each line of system results into usable information"""
        pa = r'([0-9]+?) 0 ([0-9]+?) ([0-9]+?) (.+?) 0'
        sys_rlt = re.findall(pa, line)
        query_num, doc_num, rank, score = sys_rlt[0]
        return query_num, doc_num, rank, score
    
    @classmethod
    def read_system_results(cls, sys_rlts_file):
        """read and parse system results file"""
        with open(sys_rlts_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        results = defaultdict(list)
        for line in lines:
            query_num, doc_num, rank, score = cls.parse_system_result(line)
            results[int(query_num)] += [(int(rank), int(doc_num), float(score))]
        print('Loading system results: {}...'.format(sys_rlts_file))
        return results
    
    def p_at_N(self, query_num, N=10):
        """calculate precision @ N for a query"""
        N_docs = self.current_system_results[query_num][:N]  # the sys results have been sorted
        relevant_doc_nums = [item[0] for item in self.relevant_docs[query_num]]
        count = 0
        for _, doc_num, _ in N_docs:
            if doc_num in relevant_doc_nums:
                count += 1
        return count / N
    
    def r_at_N(self, query_num, N=50):
        """calculate recall @ N for a query"""
        N_docs = self.current_system_results[query_num][:N]  # the sys results have been sorted
        relevant_doc_nums = [item[0] for item in self.relevant_docs[query_num]]
        count = 0
        for _, doc_num, _ in N_docs:
            if doc_num in relevant_doc_nums:
                count += 1
        return count / len(relevant_doc_nums)
    
    def r_precision(self, query_num):
        """calculate precision @ rank for a query"""
        rank = len(self.relevant_docs[query_num])
        return self.p_at_N(query_num, N=rank)
    
    def average_precision(self, query_num):
        """calculate AP for a query"""
        relevant_doc_nums = [item[0] for item in self.relevant_docs[query_num]]
        precisions = []
        retrieved = 0
        N = 0  # number of documents being seen
        for _, doc_num, _ in self.current_system_results[query_num]:
            N += 1
            if doc_num in relevant_doc_nums:
                retrieved += 1
                precisions.append(retrieved / float(N))  # precision at N
            # early stopping if all retrieved
            if retrieved == len(relevant_doc_nums):
                break          
        # note that when the retrieved documents do not contain all the relevant documents
        # the precision of these missing documents should be zero, and contribute to the AP,
        # that is why the denominator is the number of all relevant documents
        return sum(precisions) / len(relevant_doc_nums)
    
    def mean_average_precision(self):
        """calculate the MAP for current system"""
        total_num_of_queries = len(self.relevant_docs)
        return np.mean(self.evaluate(self.average_precision, total_num_of_queries))
    
    def DCG_at_k(self, query_num, K=10):
        """calculate the Discounted Cumulative Gain for a query"""
        relevant_inds = []
        K_docs = self.current_system_results[query_num][:K]
        inds_n_grades = []  # indices and grades of relevant documents in the top K results
        for i in range(K):
            _, doc_num, _ = K_docs[i]
            for relevant_doc_num, relevance_grade in self.relevant_docs[query_num]:
                # early stopping if found a relevant document
                if doc_num == relevant_doc_num:
                    inds_n_grades.append((i+1, relevance_grade))
                    break
        # function for calculating dcg for i > 1
        DG = lambda i, grade: grade / np.log2(i)
        # if the first index is 1
        if inds_n_grades and inds_n_grades[0][0] == 1:
            return inds_n_grades[0][1] + np.sum([DG(i, grade) for i, grade in inds_n_grades[1:]])
        else:
            return np.sum([DG(i, grade) for i, grade in inds_n_grades])
        
    def iDCG_at_k(self, query_num, K=10):
        """calculate the ideal DCG for a query"""
        # function for calculating dcg for i > 1
        DG = lambda i, grade: grade / np.log2(i)
        K_relevant_docs = self.relevant_docs[query_num][:K]
        result = K_relevant_docs[0][1]
        for i in range(1, len(K_relevant_docs)):
            grade = K_relevant_docs[i][1]
            result += DG(i+1, grade)
        return result
    
    def nDCG_at_k(self, query_num, K=10):
        """calculate the normalized DCG for a query"""
        return self.DCG_at_k(query_num, K=K) / self.iDCG_at_k(query_num, K=K)
                    
    
    def evaluate(self, metric, total_num_of_queries, arg=None):
        """evaluate all queries using a specified metric for the current system, 
           the extra args provides additional argument for the metric method"""
        if arg:
            return [metric(query_num + 1, arg) for query_num in range(total_num_of_queries)]
        else:
            return [metric(query_num + 1) for query_num in range(total_num_of_queries)]
        
    def evaluate_all(self, total_num_of_queries):
        """evaluate all queries using all required metrics for the current system"""
        df = pd.DataFrame(columns=['P@10', 'R@50', 'r-Precision', 'AP', 'nDCG@10', 'nDCG@20'])
        df['P@10'] = self.to_decimal(self.evaluate(self.p_at_N, total_num_of_queries, arg=10))
        df['R@50'] = self.to_decimal(self.evaluate(self.r_at_N, total_num_of_queries, arg=50))
        df['r-Precision'] = self.to_decimal(self.evaluate(self.r_precision, total_num_of_queries))
        df['AP'] = self.to_decimal(self.evaluate(self.average_precision, total_num_of_queries))
        df['nDCG@10'] = self.to_decimal(self.evaluate(self.nDCG_at_k, total_num_of_queries, arg=10))
        df['nDCG@20'] = self.to_decimal(self.evaluate(self.nDCG_at_k, total_num_of_queries, arg=20))
        df.index = [i+1 for i in range(10)]
        df.loc['mean'] = self.to_decimal(df.mean())
        return df
    
    @staticmethod
    def to_decimal(list_of_nums, prec=30):
        """change the list_of_nums to 3 decimal places"""
        getcontext().prec = prec  # set precision to some large number...
        return [Decimal(num).quantize(Decimal('0.000')) for num in list_of_nums]

    def output_eval_results(self, total_num_of_queries, current_system_num):
        """generate evaluation results in a output file for current system"""
        df = self.evaluate_all(total_num_of_queries)
        print(df)
        with open('S' + str(current_system_num) + '.eval', 'w+', encoding='utf-8') as f:
            f.write('\t' + '\t'.join(['P@10', 'R@50', 'r-Precision', 'AP', 'nDCG@10', 'nDCG@20']) + '\n')
            for i, dp in df.iterrows():
                f.write('\t'.join([str(i)] + [str(d) for d in dp]) + '\n')
                
    def output_eval_results_all(self, total_num_of_queries):
        """generate evaluation results in a output file for all systems"""
        return

                
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--relevant_docs_file', 
                        type=str, help='path to qrels.txt')
    parser.add_argument('--system_results',
                        type=str, help='path to system results')
    parser.add_argument('--all',
                        type=str, help='determine whether generate individual result or all')
    args = parser.parse_args()
    print(args.all)
    if not bool(args.all):
        sys_num = re.findall('S([0-9]+?).results', args.system_results)[0]
        e = Eval(relevant_docs_file = args.relevant_docs_file)
        e.current_system_results = e.read_system_results(args.system_results)
        e.output_eval_results(10, int(sys_num))