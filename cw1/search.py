import argparse
import re
import pickle
import math
from collections import deque, OrderedDict
from nltk.stem import PorterStemmer


class Search:
    """A class that implements:
            1. Query Parser (using the *Shunting Yard Algorithm*)
            2. Singleton Search (using the *Linear Merge Algorithm*):
                (a) Single-term search;
                (b) Proximity search (two terms);
                (c) Phrasal search (two terms).
            3. Boolean Search (AND, OR, NOT)
            4. IR ranking based on TF-IDF
    """


    def __init__(self, index_path, stop_words_path, stemmer):
        with open(index_path, 'rb') as f:
            self.index = pickle.load(f)
        self.stemmer = stemmer
        doc_ids = []
        for term in self.index.keys():
            doc_ids += list(self.index[term].keys())
        self.doc_ids = set(doc_ids)    # important for NOT search
        self.N = len(self.doc_ids)
        self.load_stop_words(stop_words_path)


    def load_stop_words(self, file):
        with open(file, 'r', encoding='utf-8-sig') as f:
            # remove \n for each stop word
            self.stop_words = [w.replace('\n', '') for w in f.readlines()]


    @staticmethod
    def shunting_yard(infix_tokens):
        """implementation of the Shunting Yard algorithm:
           convert infix expression to postfix expression

        Args:
            infix_tokens (str): the infix expression as a list of tokens

        Returns:
            postfix_tokens (list): the corresponding postfix expression as a list of tokens

        """

        # define precedences
        precedence = {'NOT': 3, 'AND': 2, 'OR': 1, '(': 0, ')': 0}

        # declare data strucures
        output = []
        operator_stack = []  # .pop() takes the last element out of the list
                             # operators stored always in ascending order of precedence
                             # because otherwise invoke .pop()

        # while there are tokens to be read
        for token in infix_tokens:

            # if left bracket
            if token == '(':
                operator_stack.append(token)

            # if right bracket, pop all operators from operator stack onto output until we hit a left bracket
            # (the right bracket always matches the nearest left bracket)
            elif token == ')':
                operator = operator_stack.pop()
                while not operator == '(':
                    # this means the operators must be used within the parenthesis
                    # note that the output discards the parenthesis because unneeded
                    output.append(operator)
                    operator = operator_stack.pop()

            # if operator, pop operators from operator stack to queue if they are of higher precedence than the current token
            elif token in precedence.keys():
                # if operator stack is not empty
                if operator_stack:
                    # check the precedence of the last operator (most recently added),
                    # i.e. the most precedent operator in the stack
                    last_operator = operator_stack[-1]
                    # while stack not empty and current token of lower precedence than ops in the stack
                    # (we must process the higher-precedence ops before adding a lower one)
                    while (operator_stack and precedence[last_operator] >= precedence[token]):
                        output.append(operator_stack.pop())
                        if (operator_stack):
                            last_operator = operator_stack[-1]

                operator_stack.append(token) # add token to stack

            # else if operands, add to output list
            else:
                output.append(token.lower())

        # while there are still operators on the stack, pop them into the queue
        while (operator_stack):
            output.append(operator_stack.pop())

        return output


    @staticmethod
    def tokenize_query(query):
        """read the query and split into tokens.
           type of queries:
               1. Boolean Query (AND, OR, NOT)
               2. Phrasal Query ("word1 word2 ...")
               3. Proximity Query (#distance(word1,word2))
               4. Combination of the above

           Args:
               query (string)

           Returns:
               tokens (list): split the tokens while preserving the Phrasal/Proximity queries
        """

        raw_tokens = query.split(' ')
        tokens = []
        i = 0
        while i < len(raw_tokens):
            rt = raw_tokens[i]
            # if a phrasal query
            if '\"' in rt:
                start = i
                next_token = ''  # init the next token by assigning an empty string
                # stop augmenting i when find the end token (with another ")
                while not '\"' in next_token:
                    i += 1
                    next_token = raw_tokens[i]
                i += 1
                end = i
                merged_token = ' '.join(raw_tokens[start: end])
                tokens.append(merged_token)
            # else if a proximity query
            elif '#' in rt:
                start = i
                next_token = rt  # init the next token by assigning the current token
                # stop augmenting i when find the end token (with right bracket)
                while not ')' in next_token:
                    i += 1
                    next_token = raw_tokens[i]
                i += 1
                end = i
                merged_token = ''.join(raw_tokens[start: end])
                tokens.append(merged_token)
            # else just add the token
            else:
                tokens.append(rt)
                i += 1

        return tokens


    def preprocess(self, tokens):
        """apply casefolding & normalization"""

        return [self.stemmer.stem(t.lower()) for t in tokens]


    def parse_query(self, query, search='Proximity'):
        """extract information from the query"""

        # if proximity query
        proximity_parse = re.findall(r'#([0-9]+?)\((.+?)\)', query)
        if search == 'Proximity':
            max_dist = int(proximity_parse[0][0])
            terms = self.preprocess(proximity_parse[0][1].split(','))
            return terms[0], terms[1], max_dist

        # if phrasal query
        phrasal_parse = re.findall(r'\"(.+?)\"', query)
        if search == 'Phrasal':
            terms = self.preprocess(phrasal_parse[0].split(' '))
            # distance allowed is exactly 1
            return terms[0], terms[1], 1

        return None


    def linear_merge(self, term1, term2, max_dist, search='Proximity'):
        """apply linear merge to the two posting lists"""

        posting_lists = [self.index[term1], self.index[term2]]
        docNums = [deque(sorted(posting_lists[0].keys())), \
                   deque(sorted(posting_lists[1].keys()))]
        results = []

        # apply linear merge
        # init the docNums pointers, cursor points to the current document, ref points the other
        cursor = 0
        ref = 1
        if docNums[1][0] < docNums[0][0]:
            cursor = 1
            ref = 0
        cur_doc_num = docNums[cursor].popleft()
        ref_doc_num = docNums[ref].popleft()
        # while the deques not empty
        while docNums[0] or docNums[1]:

            if cur_doc_num > ref_doc_num:
                # swap if current document ID is larger
                cursor, ref = ref, cursor
                cur_doc_num, ref_doc_num = ref_doc_num, cur_doc_num

            # move the cursor along docNums[cursor] until cur_doc_num >= ref_doc_num
            while cur_doc_num < ref_doc_num and docNums[cursor]:
                cur_doc_num = docNums[cursor].popleft()
            # marginal case: docsNums[cursor] run out but cannot find larger doc_num
            if cur_doc_num < ref_doc_num and not docNums[cursor]:
                break

            if cur_doc_num == ref_doc_num:
                # do the search if find the same doc ID on both sides
                left_term_pos = posting_lists[0][cur_doc_num]
                right_term_pos = posting_lists[1][cur_doc_num]
                # calculate the differences of positions (ref_pos - cur_pos)
                dists = [j - i for i in left_term_pos for j in right_term_pos]
                # print(dists)
                # if Proximity search, order doesn't matter
                if search == 'Proximity':
                    # using absolute value because order doesn't matter here
                    if any([abs(dist) <= max_dist for dist in dists]):
                        # print('[Success]: Find a relevant document by proximity [{}] {}'.format(max_dist, cur_doc_num))
                        results.append(cur_doc_num)
                # else if Phrasal search, order matters
                elif search == 'Phrasal':
                    # only +1 counts because the order matters
                    assert max_dist == 1
                    if max_dist in dists:
                        # print('[Success]: Find a relevant document with phrase [{}] {}'.format(" ".join([term1, term2]), cur_doc_num))
                        results.append(cur_doc_num)
                else:
                    print('[Warning!]: Check the search type!')
                # move the cursor forward, resulting next iteration must be cur > ref and do the swapping
                if docNums[cursor]:
                    cur_doc_num = docNums[cursor].popleft()
                else:
                    # marginal case: if the cursor moves to the end, not necessary to traverse the rest of the refs
                    # cause they are always bigger
                    break


        return results


    def existing(self, *words):
        """check if all the input words exist in the database"""
        for word in words:
            if not word in list(self.index.keys()):
                return False
        return True


    def singleton_search(self, query):
        """apply single-term/proximity/phrasal search to the input singleton query

           Returns:
                 [relevant documents' IDs]
        """

        results = []
        if '#' in query:
            term1, term2, max_dist = self.parse_query(query, search='Proximity')
            if self.existing(term1, term2):
                results = self.linear_merge(term1, term2, max_dist, search='Proximity')
        elif '\"' in query:
            term1, term2, max_dist = self.parse_query(query, search='Phrasal')
            if self.existing(term1, term2):
                results = self.linear_merge(term1, term2, max_dist, search='Phrasal')
        else:
            # for a single-term search, just return every document that contains it
            term = self.preprocess([query])[0]
            if self.existing(term):
                results = sorted(list(self.index[term].keys()))

        return results


    def boolean_search(self, query):
        """apply general boolean search to the input query

           Returns:
                 [relevant documents' IDs]
        """

        results_stack = []
        postfix_queue = deque(Search.shunting_yard(Search.tokenize_query(query)))
        # print('Coverse to postfix expression: ', list(postfix_queue))
        while postfix_queue:
            token = postfix_queue.popleft()
            if not token in ['AND', 'OR', 'NOT']:
                results_stack.append(self.singleton_search(token))
            elif token == 'AND':
                right = set(results_stack.pop())
                left = set(results_stack.pop())
                # left and right
                results_stack.append(sorted(list(left.intersection(right))))
            elif token == 'OR':
                right = set(results_stack.pop())
                left = set(results_stack.pop())
                # left or right
                results_stack.append(sorted(list(left.union(right))))
            elif token == 'NOT':
                right = set(results_stack.pop())
                results_stack.append(sorted(list(self.doc_ids.difference(right))))

        # at this stage, the results_stack should cotain only one list
        assert len(results_stack) == 1
        return results_stack[0]


    def boolean_search_multiple(self, query_path):
        """apply general boolean search to the multiple queries
           contained in the input file of the format:
               QueryID Query
               QueryID Query
               ...

           Args:
               query_path: the path to the input file containing queries

           Outputs:
               results.boolean.txt
        """

        with open(query_path, 'r', encoding='utf-8-sig') as f:
            queries = f.readlines()

        with open('results.boolean.txt', 'w+', encoding='utf-8') as o:
            pa = r'([0-9]+?) (.+?)\n'  # the regex of each query
            # preprocess each query
            for query in queries:
                queryID, query = re.findall(pa, query)[0]
                print('------Processing the query {}------'.format(queryID))
                results = self.boolean_search(query)  # a list of relevant document ids
                for doc_id in results:
                    o.write('{} 0 {} 0 1 0\n'.format(queryID, doc_id))


    """ ---------- Below implements the IR by ranking based on TF-IDF ---------- """
    def _tf(self, term, docID):
        """calculate the term frequency of a paricular term in a particular document"""
        # term = self.stemmer.stem(term.lower())
        return len(self.index[term][docID])


    def _df(self, term):
        """calculate the document frequency of a particular term"""
        # term = self.stemmer.stem(term.lower())
        return len(self.index[term].keys())


    def _weight(self, term, docID):
        """calculate the weight assigned to a particular term given a particular document:
           Formula:
                𝑤_{𝑡, 𝑑} = (1 + 𝑙𝑜𝑔_10(𝑡𝑓(𝑡, 𝑑))) × 𝑙𝑜𝑔_10(N / df(t))
        """
        tf = self._tf(term, docID)
        df = self._df(term)
        N = self.N
        return (1 + math.log10(tf)) * math.log10(N / float(df))


    def _tokenize(self, query):
        """apply same tokenization as in building the index
           the difference is this time we do not need the stop_word list as we have our index ready
        """
        tokens = re.findall(r'\w+', query)
        tokens = [self.stemmer.stem(t.lower()) for t in tokens \
                    if not t.lower() in self.stop_words]
        return tokens


    def _score(self, query, docID):
        """calculate the retrieval score of a query w.r.t a document
           Formula:
               Score(q, d) = \sum_(t \in q and t \in d) w_{t, d}
        """
        tokens = self._tokenize(query)
        score = 0
        for term in tokens:
            # check whether or not the document contains the current term
            # otherwise tf will be zero, rendering exception in log10
            if docID in self.index[term]:
                score += self._weight(term, docID)

        return score


    def _extract_relevant(self, query):
        """extract relevant documents of terms in the query"""
        tokens = self._tokenize(query)
        return sorted(list(set().union(*(list(self.index[t].keys()) for t in tokens))))


    def _ranking(self, query):
        """give rankings of documents based on Score(q, d)"""
        scores_dict = dict()
        # compute the relevant documents by taking the union
        for docID in self._extract_relevant(query):
            scores_dict[docID] = self._score(query, docID)
        return OrderedDict(sorted(scores_dict.items(), key=lambda kv: kv[1], reverse=True))


    def ranked_retrieval(self, query_path, max_keep=1000):
        """generate results.ranked.txt"""

        with open(query_path, 'r', encoding='utf-8') as f:
            queries = f.readlines()

        with open('results.ranked.txt', 'w+', encoding='utf-8') as r:
            pa = r'([0-9]+?) (.+?)\n'  # the regex of each query
            for query in queries:
                queryID, query = re.findall(pa, query)[0]
                print('------Generating the rankings for the query {}------'.format(queryID))
                print(query)
                ranked_dict = self._ranking(query)
                count = 0  # trace how many examples kept for each
                for docID, score in ranked_dict.items():
                    if count < max_keep:
                        r.write("{} 0 {} 0 {:.4f} 0\n".format(queryID, docID, score))
                        count += 1
                    else:
                        break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=str, help="the path to the binary index")
    parser.add_argument("--search", type=str, help="the search type: {bool, rank}")
    parser.add_argument("--query", type=str, help="the path to the query file")
    parser.add_argument("--st", type=str, help="the path to the stop-words list")
    args = parser.parse_args()

    search = Search(args.index, args.st, PorterStemmer())
    if args.search == 'bool':
        search.boolean_search_multiple(args.query)
    elif args.search == 'rank':
        search.ranked_retrieval(args.query)
