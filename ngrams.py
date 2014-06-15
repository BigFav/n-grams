#!/usr/bin/env python
from __future__ import division, unicode_literals
import argparse
import random
import re
import string
import sys
from copy import copy
from math import log10, isnan
from collections import Counter, OrderedDict, defaultdict, deque


"""
@description    n-grams, perplexity, and perplexity-based classification
@author         Favian Contreras <fnc4@cornell.edu>
"""


class Wrapper:
    """
    Wrapper created to pass by reference into a tuple, etc.
    While faster than using a list, I only did it as an exercise
    of overriding built-ins (as it only shows as faster when done
    over a million times).
    """
    def __init__(self):
        self.datum = None

    def __getitem__(self, index):
        return self.datum[index]

    def __iter__(self):
        return iter(self.datum)

    def __len__(self):
        return len(self.datum)

    def pop(self, index):
        return self.datum.pop(index)

    def set_datum(self, datum):
        self.datum = datum


class Ngrams:
    def __init__(self, opts):
        self.threshold = opts.threshold  # freq of words considered unknown
        self.start_token = "<s>"
        self.end_token = "</s>"
        self.unk_token = "<u>"
        self.bg_tks = None
        self.unigrams = None
        self.uni_ocm = None
        self.bigrams = None  # Keep separate for potential backoff
        self.bi_ocm = None
        self.ngrams = None
        self.total_words = None  # Don't calc everytime, store and use later
        self.train_len = None
        self.types = None
        self.alpha = opts.laplace
        self.n = opts.n
        self.num_features = None
        self.feature_num = opts.classify
        self.training_set = opts.training_set
        self.test_set = opts.test_set
        self.python2 = sys.version_info < (3,)

    def error_handler(self, error_code):
        print("Error: " + {
            3: ("1st line of the file doesn't define categories; cannot "
                "classify. Ex: 'category1, category2 ,\u2026,text'."),
            4: "Attempting to classify category that is not defined.",
            5: "More categories are defined than are present.",
            6: "Category only has one class in the training set."
        }[error_code])
        sys.exit(error_code)

    def init(self, n, unk, string=None):
        self.bigrams = None
        self.bi_ocm = None
        self.total_words = None

        tokens, string = self.processFile(n, 0, string)

        if n == 1:
            word_freq_pairs = self.uni_count_pairs(tokens, n, unk)
        elif n == 2:
            word_freq_pairs = self.bi_count_pairs(tokens, n, unk)
        else:
            word_freq_pairs = self.n_count_pairs(tokens, n, unk)

        self.types = len(word_freq_pairs) * self.alpha
        return string, word_freq_pairs, self.total_words

    def parse_file(self, n, typ):
        filename = self.test_set if typ % 2 else self.training_set
        if self.python2:
            range_wrap = xrange
            punct = (string.punctuation.replace("?", "").replace("'", "").
                                        replace("!", "").replace(".", ""))
            with open(filename, 'r') as text:
                tokens = unicode(text.read(), errors='replace')

        else:
            range_wrap = range
            punct = string.punctuation.translate(str.maketrans(
                                                 "", "", ".?!'"))
            with open(filename, 'r', errors="replace") as text:
                tokens = text.read()

        if n == 1:
            tokens = tokens.lower()

        # Ensure these tokens aren't in the text
        while self.start_token in tokens:
            self.start_token += '>'
        while self.end_token in tokens:
            self.end_token += '>'
        while self.unk_token in tokens:
            self.unk_token += '>'

        begin_tokens = ""
        for i in range_wrap(n-1):
            begin_tokens += ' ' + self.start_token
        self.bg_tks = begin_tokens

        start_tokens = ' ' + self.end_token
        start_tokens += begin_tokens

        # General file processing, add spaces, use spec unicode chars, etc.
        tokens = tokens.replace(":)", " \u1F601 ")
        for ch in punct:
            tokens = tokens.replace(ch, ' ' + ch + ' ')
        tokens = re.sub("\.\.+", " \u2026 ", tokens)
        tokens = tokens.replace(".", " ." + start_tokens)
        tokens = re.sub("(\!+\?|\?+\!)[?!]*", " \u203D" + start_tokens, tokens)
        tokens = re.sub("\!\!+", " !!" + start_tokens, tokens)
        tokens = re.sub("\?\?+", " ??" + start_tokens, tokens)
        tokens = re.sub("(?<![?!\s])([?!])", r" \1" + start_tokens, tokens)
        tokens = re.sub("(?<=[a-zI])('[a-z][a-z]?)\s", r" \1 ", tokens)
        if self.feature_num:
            # Ensure every line ends with an end token
            tokens = re.sub("(?<!%s)\n(?!$)" % self.start_token,
                            " %s\n" % start_tokens, tokens)

            # Find the number of categories of classes
            comas = self.num_features = tokens.count(',', 0, tokens.find('\n'))
            if not comas:
                self.error_handler(3)
            if comas < self.feature_num:
                self.error_handler(4)
            non_commas = "[^,]+," * comas

            # Insert the start tokens after these classes
            tokens = re.sub("%s\n(%s)" % (begin_tokens, non_commas),
                            r"\n\1" + begin_tokens, tokens)
        else:
            # Treat as a block of text
            tokens = re.sub("\n(?=[^\n])", " ", tokens)

        if self.end_token not in tokens:
            tokens += start_tokens if n > 1 else self.end_token

        return tokens

    def processFile(self, n, typ, string=None):
        if self.python2:
            list_wrap = lambda x: x
            range_wrap = xrange
        else:
            list_wrap = list
            range_wrap = range

        if not string:
            string = self.parse_file(n, typ)

        if typ < 2:
            # Put the leftover start tokens in the beginning
            tmp = string.split()
            tokens = []
            for i in range_wrap(n-1):
                tokens.append(tmp.pop())
            tokens.extend(tmp)

            if not typ:
                self.train_len = len(tokens)
            return tokens, string

        tokens = list_wrap(filter(bool, string.split('\n')))
        tokens[-1] = tokens[-1].strip()
        # Put the leftover start tokens in the beginning
        if tokens[-1][-len(self.start_token):] == self.start_token:
            tokens[-1] = tokens[-1][:-len(self.bg_tks)]
        else:
            tokens[-1] += ' ' + self.end_token

        num_features = tokens[0].count(',')
        del tokens[0]

        if num_features != self.num_features:
            print("Warning: The sets do not match. One has more categories "
                  "than the other.")
        last_pos = -1
        for _ in range_wrap(num_features):
            last_pos = tokens[0].find(',', last_pos+1)
            if last_pos == -1:
                self.error_handler(5)

        last_pos += 1
        tokens[0] = tokens[0][:last_pos] + self.bg_tks + tokens[0][last_pos:]

        if typ != 2:
            return tokens

        class_sets = defaultdict(list)
        for line in tokens:
            # Find the commas around the class number
            end_comma = -1
            for _ in range_wrap(self.feature_num):
                begin_comma = end_comma
                end_comma = line.find(',', end_comma+1)
                if end_comma == -1:
                    self.error_handler(5)

            # Find last comma in the line
            last_comma = end_comma
            for _ in range_wrap(self.num_features-self.feature_num):
                last_comma = line.find(',', last_comma+1)
                if last_comma == -1:
                    self.error_handler(5)

            clas = line[begin_comma+1:end_comma].strip()
            class_sets[clas].extend(line[last_comma+2:].split())

        if len(class_sets) == 1:
            self.error_handler(6)
        return class_sets

    """
    Get total counts, and word frequency dictionaries.
    """
    def uni_count_pairs(self, tokens, n, unk):
        self.total_words = len(tokens)
        word_freq_pairs = dict.fromkeys(tokens, 0)
        word_freq_pairs[self.unk_token] = 0

        for token in tokens:
            word_freq_pairs[token] += 1

        if unk:
            unk_words = set()
            items = (word_freq_pairs.iteritems() if self.python2 else
                     word_freq_pairs.items())
            for word, count in items:
                if count <= self.threshold:
                    unk_words.add(word)
                    word_freq_pairs[self.unk_token] += count

            unk_words.discard(self.start_token)
            unk_words.discard(self.end_token)
            unk_words.discard(self.unk_token)

            for word in unk_words:
                del word_freq_pairs[word]

        return word_freq_pairs

    def top_lvl_unk_tokenize(self, total_words, tokens):
        unk_words = set()
        itms = total_words.iteritems() if self.python2 else total_words.items()
        for word, count in itms:
            if count <= self.threshold:
                unk_words.add(word)
                total_words[self.unk_token] += count

        unk_words.discard(self.start_token)
        unk_words.discard(self.end_token)
        unk_words.discard(self.unk_token)

        if unk_words:
            tokens = [self.unk_token if word in unk_words else word
                      for word in tokens]
        return tokens, unk_words

    def bottom_unk_tokenize(self, word_freq_pairs, n):
        list_wrap = list if not self.python2 else lambda x: x
        tmp_pairs = word_freq_pairs
        stack = [(tmp_pairs, n)]

        while stack:
            tmp_pairs, n = stack.pop()
            if n == 2:
                values = (tmp_pairs.itervalues() if self.python2 else
                          tmp_pairs.values())
                for nxt_lvl_dict in values:
                    for word, count in list_wrap(nxt_lvl_dict.items()):
                        if (count <= self.threshold and
                                word != self.unk_token):
                            del nxt_lvl_dict[word]
                            nxt_lvl_dict[self.unk_token] += count
            else:
                n -= 1
                for word in tmp_pairs:
                    stack.append((tmp_pairs[word], n))

        return word_freq_pairs

    def bi_count_pairs(self, tokens, n, unk):
        list_wrap = (lambda x: x) if self.python2 else list
        start_token = self.start_token
        end_token = self.end_token
        unk_token = self.unk_token
        thresh = self.threshold

        self.total_words = Counter(tokens)
        self.total_words[self.end_token] -= 1  # Last token won't have a bigram
        self.total_words[self.unk_token] = 0

        if unk:
            tokens, unks = self.top_lvl_unk_tokenize(self.total_words, tokens)
            for word in unks:
                del self.total_words[word]

        word_freq_pairs = {word: defaultdict(int) for word in self.total_words}
        for i, token in enumerate(tokens[:-1]):
            word_freq_pairs[token][tokens[i+1]] += 1

        return (self.bottom_unk_tokenize(word_freq_pairs, self.n) if unk else
                word_freq_pairs)

    """
    Increments the counts for the dictionaries (used to create the
    dicts). Takes in the dicts so far, and the current word, along
    with the next n-1 words. O(1) space extra space.
    """
    def dict_creator(self, freq_dict, wrd, words):
        if words:
            freq_tmp = freq_dict
            count_tmp = self.total_words[wrd]

            # Walk through the dicts with the words
            for word in words[:-3]:
                if not freq_tmp or not freq_tmp[word]:
                    freq_tmp[word] = defaultdict(dict)
                if not count_tmp or not count_tmp[word]:
                    count_tmp[word] = defaultdict(dict)
                freq_tmp = freq_tmp[word]
                count_tmp = count_tmp[word]

            # Increment the counts
            if self.n > 3:
                if not count_tmp or not count_tmp[words[-3]]:
                    count_tmp[words[-3]] = defaultdict(int)
                count_tmp[words[-3]][words[-2]] += 1

                if not freq_tmp or not freq_tmp[words[-3]]:
                    freq_tmp[words[-3]] = defaultdict(dict)
                freq_tmp = freq_tmp[words[-3]]
            else:
                count_tmp[words[-2]] += 1

            if not freq_tmp or not freq_tmp[words[-2]]:
                freq_tmp[words[-2]] = defaultdict(int)
            freq_tmp[words[-2]][words[-1]] += 1


    def n_count_pairs(self, tokens, n, unk):
        if unk:
            # Replace low-freq top-level tokens with unks
            self.total_words = Counter(tokens)
            self.total_words[self.unk_token] = 0
            tokens, _ = self.top_lvl_unk_tokenize(self.total_words, tokens)

        dict_type = dict if n > 3 else int
        self.total_words = {token: defaultdict(dict_type) for token in tokens}
        word_freq_pairs = {token: defaultdict(dict) for token in tokens}

        words_infront = []
        for word in tokens[1:self.n]:
            words_infront.append(word)

        # Count the ngrams as reading the tokens
        for i, token in enumerate(tokens[:-self.n]):
            self.dict_creator(word_freq_pairs[token], token, words_infront)
            del words_infront[0]
            words_infront.append(tokens[i+self.n])
        token = tokens[-self.n]
        self.dict_creator(word_freq_pairs[token], token, words_infront)

        return (self.bottom_unk_tokenize(word_freq_pairs, self.n) if unk else
                word_freq_pairs)

    """
    Computes MLE probability distributions.
    """
    def unsmoothed_unigrams(self, word_freq_pairs):
        prob_dict = word_freq_pairs
        items = prob_dict.iteritems() if self.python2 else prob_dict.items()
        for word, count in items:
            prob_dict[word] = count / self.total_words

        self.unigrams = prob_dict

    def unsmoothed_bigrams(self, word_freq_pairs):
        prob_dict = word_freq_pairs
        items = prob_dict.iteritems() if self.python2 else prob_dict.items()
        for word, nxt_lvl_dict in items:
            nxt_lvl_items = (nxt_lvl_dict.iteritems() if self.python2 else
                             nxt_lvl_dict.items())
            for word_infront, cnt in nxt_lvl_items:
                nxt_lvl_dict[word_infront] = cnt / self.total_words[word]

        self.bigrams = prob_dict

    def unsmoothed_ngrams(self, word_freq_pairs, total_words, n):
        prob_dict = word_freq_pairs
        if n == 2:
            items = (prob_dict.iteritems() if self.python2 else
                     prob_dict.items())
            for word, nxt_lvl_dict in items:
                nxt_lvl_items = (nxt_lvl_dict.iteritems() if self.python2 else
                                 nxt_lvl_dict.items())
                for word_infront, count in nxt_lvl_items:
                    nxt_lvl_dict[word_infront] = count / total_words[word]
            return

        for word in prob_dict:
            self.unsmoothed_ngrams(prob_dict[word], total_words[word], n - 1)

        self.ngrams = prob_dict

    """
    Computes Laplace smoothed probability distributions.
    """
    def laplace_unigrams(self, word_freq_pairs, total_words, V):
        prob_dict = word_freq_pairs
        items = prob_dict.iteritems() if self.python2 else prob_dict.items()
        for word, count in items:
            prob_dict[word] = (count+self.alpha) / (total_words+V)

        self.unigrams = prob_dict

    def laplace_ngrams(self, word_freq_pairs, total_words, n, V):
        alpha = self.alpha
        stack = [(word_freq_pairs, total_words, n)]
        prob_dict = word_freq_pairs

        # Iterative walk using "stack," marginally faster than recursion
        # Iterative walk using "queue" is much slower due to struct of dicts
        while stack:
            word_freq_pairs, total_words, my_n = stack.pop()
            if my_n == 2:
                items = (word_freq_pairs.iteritems() if self.python2 else
                         word_freq_pairs.items())
                for top_word, nxt_lvl_dict in items:
                    nxt_lvl_items = (nxt_lvl_dict.iteritems() if self.python2
                                     else nxt_lvl_dict.items())
                    for bot_word, cnt in nxt_lvl_items:
                        nxt_lvl_dict[bot_word] = ((cnt+alpha) /
                                                  (total_words[top_word]+V))
            else:
                my_n -= 1
                for word in word_freq_pairs:
                    stack.append((word_freq_pairs[word],
                                  total_words[word], my_n))
        if n == 2:
            self.bigrams = prob_dict
        else:
            self.ngrams = prob_dict

    """
    Creates a dict of how many times a word of a certain frequency occurs.
    Then gets probabilty distributions from good turing smoothing.
    """
    def occurrenceToUniTuring(self, word_freq_pairs, total_words):
        if self.python2:
            range_wrap = xrange
            values = word_freq_pairs.itervalues()
        else:
            range_wrap = range
            values = word_freq_pairs.values()

        occurence_map = OrderedDict.fromkeys(range_wrap(1, max(values)+2), 0)

        values = (word_freq_pairs.itervalues() if self.python2 else
                  word_freq_pairs.values())
        for value in values:
            occurence_map[value] += 1
        if word_freq_pairs[self.unk_token] <= self.threshold:
            occurence_map[word_freq_pairs[self.unk_token]] = 1

        #fill in the levels with 0 words
        last_val = 1
        list_wrap = (lambda x: x) if self.python2 else list
        for key, value in reversed(list_wrap(occurence_map.items())):
            if not value:
                occurence_map[key] = last_val
            last_val = occurence_map[key]
        self.uni_ocm = occurence_map

        self.goodTuringSmoothUni(word_freq_pairs, occurence_map, total_words)

    def goodTuringSmoothUni(self, word_freq_pairs, uni_ocm, total_words):
        prob_dict = word_freq_pairs
        items = prob_dict.iteritems() if self.python2 else prob_dict.items()
        for word, count in items:
            prob_dict[word] = ((count+1) * uni_ocm[count+1] / uni_ocm[count] /
                               total_words)

        self.unigrams = prob_dict

    def occurrenceToBiTuring(self, word_freq_pairs, total_words):
        if self.python2:
            list_wrap = lambda x: x
            range_wrap = xrange
            keys = word_freq_pairs.iterkeys()
        else:
            list_wrap = list
            range_wrap = range
            keys = word_freq_pairs.keys()

        unk_token = self.unk_token
        occurence_map = dict.fromkeys(keys)

        items = (word_freq_pairs.iteritems() if self.python2 else
                 word_freq_pairs.items())
        for wrd, nxt_lvl_dict in items:
            if nxt_lvl_dict:
                nxt_lvl_vals = (nxt_lvl_dict.itervalues() if self.python2 else
                                nxt_lvl_dict.values())
                top = max(nxt_lvl_vals)
                occurence_map[wrd] = OrderedDict.fromkeys(range_wrap(1, top+2),
                                                          0)

                nxt_lvl_vals = (nxt_lvl_dict.itervalues() if self.python2 else
                                nxt_lvl_dict.values())
                for value in nxt_lvl_vals:
                    occurence_map[wrd][value] += 1
            else:
                # "Fill" with unk_token, if empty
                occurence_map[wrd] = {1: 1}
                self.total_words[wrd] = 1

            # Fills in the levels with 0 words
            last_val = 1
            for occur, cnt in reversed(list_wrap(occurence_map[wrd].items())):
                if not cnt:
                    occurence_map[wrd][occur] = last_val
                else:
                    last_val = occurence_map[wrd][occur]

        self.bi_ocm = occurence_map
        self.goodTuringSmoothBi(word_freq_pairs, total_words, occurence_map)

    def goodTuringSmoothBi(self, word_freq_pairs, total_words, bi_ocm):
        prob_dict = word_freq_pairs
        items = prob_dict.iteritems() if self.python2 else prob_dict.items()
        for w, nxt_lvl_dict in items:
            nxt_lvl_items = (nxt_lvl_dict.iteritems() if self.python2 else
                             nxt_lvl_dict.items())
            for w_infront, cnt in nxt_lvl_items:
                nxt_lvl_dict[w_infront] = ((cnt+1) * bi_ocm[w][cnt+1] /
                                           bi_ocm[w][cnt] / total_words[w])
        self.bigrams = prob_dict

    """
    Generates sentences based on probability distributions.
    """
    def generateSentence(self, n):
        sentence = []
        words = [self.start_token] * (n-1)
        if n == 1:
            ngrams = self.unigrams
        elif n == 2:
            ngrams = self.bigrams
        else:
            ngrams = self.ngrams

        word = self.weightedPickN(words, ngrams)
        while word != self.end_token:
            if n != 1:
                del words[0]
                words.append(word)
            sentence.append(word)
            word = self.weightedPickN(words, ngrams)

        print(' '.join(sentence))

    def weightedPickN(self, words, tmp_dict):
        for word in words:
            try:
                tmp_dict = tmp_dict[word]
            except KeyError:
                return self.end_token

        s = 0.0
        key = ""
        values = tmp_dict.itervalues() if self.python2 else tmp_dict.values()
        r = random.uniform(0, sum(values))
        items = tmp_dict.iteritems() if self.python2 else tmp_dict.items()
        for key, weight in items:
            s += weight
            if r < s:
                return key
        return key

    def uni_perplex(self, tokens, gts, unigrams=None,
                    train_len=None, uni_ocm=None, V=None):
        if not unigrams:
            unigrams = self.unigrams
            train_len = self.train_len

        entropy = 0.0
        if gts:
            if not uni_ocm:
                uni_ocm = self.uni_ocm
            thresh = self.threshold
            for token in tokens:
                entropy -= log10(unigrams.get(token, uni_ocm[thresh] /
                                                     train_len))
        else:
            if not V:
                V = self.types
            alpha = self.alpha
            for token in tokens:
                entropy -= log10(unigrams.get(token, alpha / (train_len+V)))

        return 10**(entropy / (len(tokens) - (self.n-1)))

    def bi_perplex(self, tokens, gts, bigram=None,
                   tw=None, bi_ocm=None, V=None):
        if not tw:
            tw = self.total_words
            bigram = self.bigrams

        ut = self.unk_token
        thresh = self.threshold
        entropy = 0.0
        prev_t = tokens.pop(0)
        if gts:
            if not bi_ocm:
                bi_ocm = self.bi_ocm
            for token in tokens:
                if prev_t in bigram:
                    entropy -= log10(bigram[prev_t].get(token,
                                     bi_ocm[prev_t][thresh] / tw[prev_t]))
                else:
                    entropy -= log10(bigram[ut].get(token,
                                     bi_ocm[ut][thresh] / tw[ut]))
                prev_t = token
        else:
            if not V:
                V = self.types
            alpha = self.alpha
            for token in tokens:
                if prev_t in bigram:
                    entropy -= log10(bigram[prev_t].get(token, alpha /
                                                        (tw[prev_t]+V)))
                else:
                    entropy -= log10(bigram[ut].get(token, alpha / (tw[ut]+V)))

                prev_t = token

        return 10**(entropy / (len(tokens) - (self.n-1)))

    """
    def sum_count(self, p):
        while p and isinstance(p[0], dict):
            for v in p:
                if isinstance(v, dict):
                    for value in v.values():
                        p.append(value)
                    del p[0]
        return sum(num.isdigit() for num in map(str, p)) if p else self.train_len
    """

    def n_laplace_perplex(self, tokens, ngrams, total_words, types, n):
        help_dict = ngrams
        if n == 1:
            return log10(help_dict.get(tokens[0], self.alpha /
                                                  (total_words+types)))

        nxt_token = tokens.popleft()
        if nxt_token in help_dict:
            return self.n_laplace_perplex(tokens, help_dict[nxt_token],
                                          total_words[nxt_token], types, n-1)
        """
        if hash(''.join(help_dict.keys())) in sum_dict:
            total = sum_dict[hash(''.join(help_dict.keys()))]
        else:
            total = self.sum_count(help_dict.values())
            sum_dict[hash(''.join(help_dict.keys()))] = total
        """
        # small "punishment" for not being in it (100), real approx is too slow
        return log10(self.alpha / (100 + types))

    def n_laplace_perplex_help(self, tokens, n,
                               ngram=None, tw=None, types=None):
        #sum_dict = {}
        if not ngram:
            types = self.types
            ngram = self.ngrams
            tw = self.total_words

        entropy = 0.0
        num_tokens = len(tokens)
        words = deque(tokens[:n])
        range_wrap = xrange if self.python2 else range
        for i in range_wrap(num_tokens - n):
            entropy -= self.n_laplace_perplex(copy(words), ngram, tw, types, n)
            del words[0]
            words.append(tokens[i+n])
        entropy -= self.n_laplace_perplex(words, ngram, tw, types, n)

        return 10**(entropy / (num_tokens - (n-1)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=2,
                        help="Size of the probability tuples. N should be < "
                             "20 and N must be >= 1.")
    parser.add_argument("-sent", "--sentence", action="store_true",
                        help="Generate sentence based on unsmoothed "
                             "n-grams of the training set.")
    parser.add_argument("-t", "--threshold", action="store", type=int,
                        default=1, metavar='T',
                        help="Set the threshold; all words that have a "
                             "frequency <= to the threshold are considered "
                             "unknown (1 by default). Threshold must be >= 1 "
                             "to perform perplexity based operations.")

    smoothing_types = parser.add_mutually_exclusive_group()
    smoothing_types.add_argument("-ls", "--laplace", action="store", nargs='?',
                                 type=float, default=1, metavar="\u03B1",
                                 const=1, help="Use Laplace (Additive) "
                                               "smoothing; plus-one is the "
                                               "default (\u03B1 = 1).")
    smoothing_types.add_argument("-gts", "--turing", action="store_true",
                                 help="Use Good Turing smoothing. Only "
                                      "available for n <= 2.")

    parser.add_argument("-p", "--perplexity", action="store_true",
                        help="Compute the perplexity of the test set, "
                             "trained on the training set.")
    parser.add_argument("-c", "--classify", metavar="CATEGORY_NUM", nargs='?',
                        default="",
                        help="Classify the text based on perplexity on "
                             "the specified category number (1 by default). "
                             "Ex: category1, category2 ,\u2026,text.")

    parser.add_argument("output_file", action="store", nargs='?')
    parser.add_argument("training_set", action="store",
                        help="Must be at end of command.")
    parser.add_argument("test_set", action="store", nargs='?',
                        help="Must be at end of command.")

    parser.usage = ("ngrams.py [-h] [-n N] -sent training_set\n              "
                    "   [-h] [-n N] [-sent] [-t T] [-ls [\u03B1] | -gts] [-p] "
                    "[-c [CATEGORY_NUM] output_file] training_set test_set")

    error_str = ""
    opts = parser.parse_args()
    if opts.classify and opts.output_file and not opts.test_set:
        opts.test_set = opts.training_set
        opts.training_set = opts.output_file
        opts.output_file = opts.classify
        opts.classify = 1  # Default
    elif (not opts.classify and opts.output_file and
          not (opts.test_set and opts.training_set)):
        # Only flip files if they are missing, otherwise ignore
        opts.test_set = opts.training_set
        opts.training_set = opts.output_file
    elif opts.classify and not opts.output_file:
        error_str += ("too few arguments: must input an output file when "
                      "performing classification (-c).\n")
    elif ((opts.classify or (opts.classify == 0)) and opts.output_file and
          opts.test_set and opts.training_set):
        try:
            opts.classify = int(opts.classify)
        except ValueError:
            error_str += ("argument -c: invalid int value: "
                          "'%s'\n" % opts.classify)
        else:
            if opts.classify <= 0:
                error_str += ("argument -c: invalid int value: "
                              "category number must be >= 1.\n")
    if opts.perplexity or opts.classify or (opts.classify == 0):
        if not opts.test_set:
            if error_str:
                error_str += "                  "
            error_str += ("too few arguments: must input a test set, if "
                          "performing perplexity based operations "
                          "(-p and -c).\n")
        if opts.threshold < 1:
            if error_str:
                error_str += "                  "
            error_str += ("argument -t: invalid int value: threshold must be "
                          ">= 1, if performing perplexity based operations "
                          "(-p and -c).\n")
    if opts.threshold < 0:
        if error_str:
            error_str += "                  "
        error_str += "argument -t: invalid int value: threshold must be > 0.\n"
    if opts.n < 1:
        if error_str:
            error_str += "                  "
        error_str += "argument -n: invalid int value: N must be >= 1.\n"
    if opts.n > 20:
        input = raw_input if sys.version_info < (3,) else input
        warning_str = ("Warning: program may crash at such a high N. "
                       "Continue? (Y/n) ")
        while True:
            cont = input(warning_str)
            if (cont == 'n') or (cont == 'N'):
                if error_str:
                    error_str += "                  "
                error_str += ("argument -n: choose to not continue "
                              "with high N.\n")
                break
            elif (cont == 'y') or (cont == 'Y'):
                break
            warning_str = "Please enter y or N: "
    if opts.turing and (opts.n > 2):
        if error_str:
            error_str += "                  "
        error_str += ("argument -n: invalid int value: Good Turing "
                      "smoothing only available for n <= 2.\n")
    if isnan(opts.laplace) or (opts.laplace == float('inf')):
        if error_str:
            error_str += "                  "
        error_str += ("argument -ls: invalid int value: \u03B1 cannot "
                      "be NaN and cannot be inf (\u221E).\n")

    if error_str:
        parser.error(error_str[:-1])
    return opts


def finish_model(model, n, gts, word_freq_pairs, total_words, types):
    if gts:
        if n == 1:
            model.occurrenceToUniTuring(word_freq_pairs, total_words)
        else:
            model.occurrenceToBiTuring(word_freq_pairs, total_words)
    else:
        if n == 1:
            model.laplace_unigrams(word_freq_pairs, total_words, types)
        else:
            model.laplace_ngrams(word_freq_pairs, total_words, n, types)


def main():
    opts = parse_args()
    n = opts.n
    gts = opts.turing
    train_str = test_str = None

    model = Ngrams(opts)
    if opts.sentence:
        train_str, word_freq_pairs, total_words = model.init(n, 0)
        if n == 1:
            model.unsmoothed_unigrams(word_freq_pairs)
        elif n == 2:
            model.unsmoothed_bigrams(word_freq_pairs)
        else:
            model.unsmoothed_ngrams(word_freq_pairs, total_words, n)

        model.generateSentence(n)

    if opts.perplexity:
        train_str, word_freq_pairs, total_words = model.init(n, 1, train_str)
        finish_model(model, n, gts, word_freq_pairs, total_words, model.types)
        test_t, test_str = model.processFile(n, 1, None)
        if n == 1:
            perplexity = model.uni_perplex(test_t, gts)
        elif n == 2:
            perplexity = model.bi_perplex(test_t, gts)
        else:
            perplexity = model.n_laplace_perplex_help(test_t, n)

        print("Perplexity: " + str(perplexity))

    if opts.classify:
        t_arg = Wrapper()
        freq_pairs = {}
        types = {}
        words = {}
        args = {}
        class_set = model.processFile(n, 2, train_str)
        if n == 1:
            unigrams = {}
            ocm = {}
            items = (class_set.iteritems() if model.python2 else
                     class_set.items())
            for clas, class_words in items:
                freq_pairs[clas] = model.uni_count_pairs(class_words, n, True)
                types[clas] = len(freq_pairs[clas]) * opts.laplace
                words[clas] = len(class_words)

                finish_model(model, n, gts,
                             freq_pairs[clas], words[clas], types[clas])
                unigrams[clas] = model.unigrams
                ocm[clas] = model.uni_ocm
                args[clas] = (t_arg, gts, unigrams[clas],
                              words[clas], ocm[clas], types[clas])

            perplex_fun = model.uni_perplex
        elif n == 2:
            bigrams = {}
            ocm = {}
            items = (class_set.iteritems() if model.python2 else
                     class_set.items())
            for clas, class_words in items:
                freq_pairs[clas] = model.bi_count_pairs(class_words, n, True)
                types[clas] = len(freq_pairs[clas]) * opts.laplace
                words[clas] = model.total_words

                finish_model(model, n, gts,
                             freq_pairs[clas], words[clas], types[clas])
                bigrams[clas] = model.bigrams
                ocm[clas] = model.bi_ocm
                args[clas] = (t_arg, gts, bigrams[clas],
                              words[clas], ocm[clas], types[clas])

            perplex_fun = model.bi_perplex
        else:
            ngrams = {}
            items = (class_set.iteritems() if model.python2 else
                     class_set.items())
            for clas, class_words in items:
                freq_pairs[clas] = model.n_count_pairs(class_words, n, True)
                types[clas] = len(freq_pairs[clas]) * opts.laplace
                words[clas] = model.total_words

                finish_model(model, n, gts,
                             freq_pairs[clas], words[clas], types[clas])
                ngrams[clas] = model.ngrams
                args[clas] = (t_arg, n, ngrams[clas], words[clas], types[clas])

            perplex_fun = model.n_laplace_perplex_help

        range_wrap = xrange if model.python2 else range
        predictions = []
        test_t = model.processFile(n, 3, test_str)
        for line in test_t:
            last_comma = -1
            for _ in range_wrap(model.num_features):
                last_comma = line.find(',', last_comma+1)
                if last_comma == -1:
                    model.error_handler(5)

            t_arg.set_datum(line[last_comma+2:].split())

            # Compare perplexities
            low = [None, float('inf')]
            items = args.iteritems() if model.python2 else args.items()
            for clas, params in items:
                perplexity = perplex_fun(*params)
                if perplexity < low[1]:
                    low = [clas, perplexity]

            if low[0][0] == '-':
                predictions.append('-' + low[0][2:])
            else:
                predictions.append(low[0])

        with open(opts.output_file, 'w') as guesses:
            guesses.write('\n'.join(predictions))

if __name__ == '__main__':
    main()
