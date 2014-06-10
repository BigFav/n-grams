from __future__ import division, unicode_literals
import argparse
import sys
import string
from copy import copy
from math import log10
from re import sub as re_sub
from random import uniform as rand_uniform
from collections import Counter, OrderedDict, defaultdict, deque


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
        self.bigrams = None
        self.bi_ocm = None
        self.ngrams = None
        self.total_words = None
        self.train_len = None
        self.types = None
        self.n = opts.n
        self.num_features = None
        self.feature_num = opts.classify
        self.opts = opts
        self.python2 = sys.version_info < (3,)

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

        self.types = len(word_freq_pairs)
        return string, word_freq_pairs, self.total_words

    def parse_file(self, n, typ):
        filename = self.opts.test_set if typ % 2 else self.opts.training_set
        if self.python2:
            range_wrap = xrange
            punctuation = string.punctuation.replace("?", "").replace("'", "")
            punctuation = punctuation.replace("!", "").replace(".", "")
            with open(filename, 'r') as text:
                tokens = unicode(text.read(), errors='replace')

        else:
            range_wrap = range
            punctuation = string.punctuation.translate(str.maketrans(
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
            begin_tokens += ' ' + self.start_token + ' '
        begin_tokens = begin_tokens[:-1]
        self.bg_tks = begin_tokens

        start_tokens = ' ' + self.end_token
        start_tokens += begin_tokens

        tokens = re_sub(":\)", " \u1F601 ", tokens)
        for ch in punctuation:
            tokens = tokens.replace(ch, ' ' + ch + ' ')
        tokens = re_sub("\.\.+", " \u2026 ", tokens)
        tokens = re_sub("(\!+\?|\?+\!)[?!]*",
                        " \u203D" + start_tokens, tokens)
        tokens = re_sub("\!\!+", " !!" + start_tokens, tokens)
        tokens = re_sub("\?\?+", " ??" + start_tokens, tokens)
        tokens = re_sub("((?<![.?!\s])[.?!])",
                        r" \1" + start_tokens, tokens)
        tokens = re_sub("(?<=[a-zI])('[a-z][a-z]?)\s", r" \1 ", tokens)
        if self.opts.classify:
            commas = self.num_features = tokens.partition('\n')[0].count(',')
            if not commas:
                sys.exit("1st line of the file doesn't define features; cannot"
                         " classify. Ex: 'feature1, feature2 ,\u2026,text'.")
            if commas < self.feature_num:
                sys.exit("Attempting to classify feature that is not defined.")
            non_commas = "[^,]+," * commas
            tokens = re_sub("%s\n(%s)" % (begin_tokens, non_commas),
                            r"\n\1" + begin_tokens, tokens)
        else:
            tokens = re_sub("\n(?=[^\n])", " ", tokens)

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
            tmp = list_wrap(filter(bool, string.split()))
            tokens = []
            for i in range_wrap(n-1):
                tokens.append(tmp.pop())
            tokens.extend(tmp)

            if not typ:
                self.train_len = len(tokens)
            return tokens, string

        string = string[-len(self.bg_tks):] + string[:-len(self.bg_tks)]
        tokens = list_wrap(filter(bool, string.split('\n')))
        del tokens[:2]

        last_pos = -1
        for _ in range_wrap(self.num_features):
            last_pos = tokens[0].find(',', last_pos+1)
            if last_pos == -1:
                sys.exit("More features are defined than are present.")
        last_pos += 1
        tokens[0] = tokens[0][:last_pos] + self.bg_tks + tokens[0][last_pos:]

        if typ != 2:
            return tokens

        class_set = set()
        for line in tokens:
            end_comma = -1
            for _ in range_wrap(self.feature_num):
                begin_comma = end_comma
                end_comma = line.find(',', end_comma+1)
                if end_comma == -1:
                    sys.exit("More features are defined than are present.")

            last_comma = end_comma
            for _ in range_wrap(self.num_features-self.feature_num):
                last_comma = line.find(',', last_comma+1)
                if last_comma == -1:
                    sys.exit("More features are defined than are present.")

            last_comma += 2
            begin_comma += 1
            clas = int(line[begin_comma:end_comma])
            if clas == 0:
                if neg_class:
                    sys.exit("Multiple negative classes. "
                             "Must be either 0 or -1, not both.")
                neg_class = 0
                negSet.extend(line[last_comma:].split())
            elif clas == -1:
                if neg_class == 0:
                    sys.exit("Multiple negative classes. "
                             "Must be either 0 or -1, not both.")
                neg_class = -1
                print line
                negSet.extend(line[last_comma:].split())
            elif clas == 1:
                posSet.extend(line[last_comma:].split())
            else:
                sys.exit("Class values must be positive (1), "
                         "and negative (0 or -1).")

        if not len(negSet) or not len(posSet):
            sys.exit("Only one class in the training set. Exiting now.")
        return negSet, posSet, str(neg_class)

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

    def bi_count_pairs(self, tokens, n, unk):
        self.total_words = Counter(tokens)
        self.total_words[self.end_token] -= 1
        self.total_words[self.unk_token] = 0

        if unk:
            unk_words = set()
            items = (self.total_words.iteritems() if self.python2 else
                     self.total_words.items())
            for word, count in items:
                if count <= self.threshold:
                    unk_words.add(word)
                    self.total_words[self.unk_token] += count

            unk_words.discard(self.start_token)
            unk_words.discard(self.end_token)
            unk_words.discard(self.unk_token)

            for word in unk_words:
                del self.total_words[word]

            # replace words in tokens with <u>
            if unk_words:
                tokens = [self.unk_token if word in unk_words else word
                          for word in tokens]

        word_freq_pairs = {word: defaultdict(int) for word in self.total_words}
        for i, token in enumerate(tokens[:-1]):
            word_freq_pairs[token][tokens[i+1]] += 1

        return word_freq_pairs

    def dict_creator(self, freq_dict, wrd, words):
        if words:
            freq_tmp = freq_dict
            count_tmp = self.total_words[wrd]
            for i, word in enumerate(words[:-3]):
                if not freq_tmp or not freq_tmp[word]:
                    freq_tmp[word] = defaultdict(dict)
                if not count_tmp or not count_tmp[word]:
                    count_tmp[word] = defaultdict(dict)
                freq_tmp = freq_tmp[word]
                count_tmp = count_tmp[word]

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

    def unk_tokenize(self, word_freq_pairs, n):
        if n == 2:
            if self.python2:
                list_wrap = lambda x: x
                values = word_freq_pairs.itervalues()
            else:
                list_wrap = list
                values = word_freq_pairs.values()

            for nxt_lvl_dict in values:
                for word, count in list_wrap(nxt_lvl_dict.items()):
                    if (count <= self.threshold) and (word != self.unk_token):
                        del nxt_lvl_dict[word]
                        nxt_lvl_dict[self.unk_token] += count
        else:
            for word in word_freq_pairs:
                self.unk_tokenize(word_freq_pairs[word], n - 1)

        return word_freq_pairs

    def n_count_pairs(self, tokens, n, unk):
        if unk:
            self.total_words = Counter(tokens)
            self.total_words[self.unk_token] = 0

            unk_words = set()
            items = (self.total_words.iteritems() if self.python2 else
                     self.total_words.items())
            for word, count in items:
                if count <= self.threshold:
                    unk_words.add(word)
                    self.total_words[self.unk_token] += count

            unk_words.discard(self.start_token)
            unk_words.discard(self.end_token)
            unk_words.discard(self.unk_token)

            #replace words in tokens with <u>
            if unk_words:
                tokens = [self.unk_token if word in unk_words else word
                          for word in tokens]

        dict_type = dict if n > 3 else int
        self.total_words = {token: defaultdict(dict_type) for token in tokens}
        word_freq_pairs = {token: defaultdict(dict) for token in tokens}

        words_infront = []
        for word in tokens[1:self.n]:
            words_infront.append(word)

        for i, token in enumerate(tokens[:-self.n]):
            self.dict_creator(word_freq_pairs[token], token, words_infront)
            del words_infront[0]
            words_infront.append(tokens[i+self.n])
        token = tokens[-self.n]
        self.dict_creator(word_freq_pairs[token], token, words_infront)

        return (self.unk_tokenize(word_freq_pairs, self.n) if unk else
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
    def laplace_unigrams(self, word_freq_pairs, total_words):
        prob_dict = word_freq_pairs
        dict_len = len(prob_dict)
        items = prob_dict.iteritems() if self.python2 else prob_dict.items()
        for word, count in items:
            prob_dict[word] = (count+1) / (total_words+dict_len)

        self.unigrams = prob_dict

    def laplace_bigrams(self, word_freq_pairs, total_words):
        prob_dict = word_freq_pairs
        dict_len = len(prob_dict)
        items = prob_dict.iteritems() if self.python2 else prob_dict.items()
        for top_word, nxt_lvl_dict in items:
            nxt_lvl_items = (nxt_lvl_dict.iteritems() if self.python2 else
                             nxt_lvl_dict.items())
            for bot_word, cnt in nxt_lvl_items:
                nxt_lvl_dict[bot_word] = ((cnt+1) /
                                          (total_words[top_word]+dict_len))
        self.bigrams = prob_dict

    def laplace_ngrams(self, word_freq_pairs, total_words, n):
        prob_dict = word_freq_pairs
        if n == 2:
            dict_len = len(prob_dict)
            items = (prob_dict.iteritems() if self.python2 else
                     prob_dict.items())
            for top_word, nxt_lvl_dict in items:
                nxt_lvl_items = (nxt_lvl_dict.iteritems() if self.python2 else
                                 nxt_lvl_dict.items())
                for bot_word, cnt in nxt_lvl_items:
                    nxt_lvl_dict[bot_word] = ((cnt+1) /
                                              (total_words[top_word]+dict_len))
            return

        for word in prob_dict:
            self.laplace_ngrams(prob_dict[word], total_words[word], n - 1)

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
        for word, nxt_lvl_dict in items:
            if nxt_lvl_dict:
                unk_words = set()
                nxt_lvl_items = (nxt_lvl_dict.iteritems() if self.python2 else
                                 nxt_lvl_dict.items())
                for second_word, count in nxt_lvl_items:
                    if count <= self.threshold:
                        unk_words.add(second_word)

                unk_words.discard(self.start_token)
                unk_words.discard(self.end_token)
                unk_words.discard(unk_token)
                nxt_lvl_dict.update((unk_token, nxt_lvl_dict[unk_token] + cnt)
                                    for wrd2, cnt in
                                    list_wrap(nxt_lvl_dict.items())
                                    if wrd2 in unk_words)

                for unk_word in unk_words:
                    del nxt_lvl_dict[unk_word]

                nxt_lvl_vals = (nxt_lvl_dict.itervalues() if self.python2 else
                                nxt_lvl_dict.values())
                top = max(nxt_lvl_vals)
                occurence_map[word] = OrderedDict.fromkeys(range_wrap(1, top+2
                                                                      ), 0)

                nxt_lvl_vals = (nxt_lvl_dict.itervalues() if self.python2 else
                                nxt_lvl_dict.values())
                for value in nxt_lvl_vals:
                    occurence_map[word][value] += 1
            else:
                # "Fill" with unk_token, if empty
                occurence_map[word] = {1: 1}
                self.total_words[word] = 1

            #fill in the levels with 0 words
            last_val = 1
            for key, value in reversed(list_wrap(occurence_map[word].items())):
                if not value:
                    occurence_map[word][key] = last_val
                last_val = occurence_map[word][key]

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
        word = self.start_token
        next_word_fun = self.weightedPickBi if n == 2 else self.weightedPickUni

        word = next_word_fun(word)
        while word != self.end_token:
            sentence.append(word)
            word = next_word_fun(word)

        print(' '.join(sentence))

    def weightedPickUni(self, _=None):
        s = 0.0
        key = ""
        values = (self.unigrams.itervalues() if self.python2 else
                  self.unigrams.values())
        r = rand_uniform(0, sum(values))

        items = (self.unigrams.iteritems() if self.python2 else
                 self.unigrams.items())
        for key, weight in items:
            s += weight
            if r < s:
                return key
        return key

    def weightedPickBi(self, word):
        s = 0.0
        key = ""
        values = (self.bigrams[word].itervalues() if self.python2 else
                  self.bigrams[word].values())
        r = rand_uniform(0, sum(values))

        items = (self.bigrams[word].iteritems() if self.python2 else
                 self.bigrams[word].items())
        for key, weight in items:
            s += weight
            if r < s:
                return key
        return key

    def generateNgramSentence(self):
        sentence = []
        words = [self.start_token] * (self.n-1)

        word = self.weightedPickN(words)
        while word != self.end_token:
            sentence.append(word)
            del words[0]
            words.append(word)
            word = self.weightedPickN(words)

        print(' '.join(sentence))

    def weightedPickN(self, words):
        tmp_dict = self.ngrams
        for word in words:
            try:
                tmp_dict = tmp_dict[word]
            except KeyError:
                return self.end_token

        s = 0.0
        key = ""
        values = tmp_dict.itervalues() if self.python2 else tmp_dict.values()
        r = rand_uniform(0, sum(values))
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
            for token in tokens:
                entropy -= log10(unigrams.get(token,
                                 uni_ocm[self.threshold] / train_len))
        else:
            if not V:
                V = self.types
            for token in tokens:
                entropy -= log10(unigrams.get(token, 1 / (train_len+V)))

        return 10**(entropy / (len(tokens) - (self.n-1)))

    def bi_perplex(self, tokens, gts, bigrams=None,
                   tw=None, bi_ocm=None, V=None):
        if not tw:
            tw = self.total_words
            bigrams = self.bigrams

        ut = self.unk_token
        thresh = self.threshold
        entropy = 0.0
        prev_t = tokens.pop(0)
        if gts:
            if not bi_ocm:
                bi_ocm = self.bi_ocm
            for token in tokens:
                if prev_t in bigrams:
                    entropy -= log10(bigrams[prev_t].get(token,
                                     bi_ocm[prev_t][thresh] / tw[prev_t]))
                else:
                    entropy -= log10(bigrams[ut].get(token,
                                     bi_ocm[ut][thresh] / tw[ut]))
                prev_t = token
        else:
            if not V:
                V = self.types
            for token in tokens:
                if prev_t in bigrams:
                    entropy -= log10(bigrams[prev_t].get(token,
                                                         1 / (tw[prev_t]+V)))
                else:
                    entropy -= log10(bigrams[ut].get(token, 1 / (tw[ut]+V)))

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
            return log10(help_dict.get(tokens[0], 1 / (total_words+types)))

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
        return log10(1 / (100 + types))

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
                        help="Size of the probability tuples.")
    parser.add_argument("-sent", "--sentence", action="store_true",
                        help="Generate sentence based on unsmoothed "
                             "n-grams of the training set.")
    parser.add_argument("-t", "--threshold", action="store", type=int,
                        default=1, metavar='T',
                        help="Set the threshold; all words that have a "
                             "frequency <= to the threshold are considered "
                             "unknown (1 by default). Threshold must be > 1 in"
                             "order to perform perplexity based operations.")

    smoothing_types = parser.add_mutually_exclusive_group()
    smoothing_types.add_argument("-ls", "--laplace", action="store_true",
                                 help="Use Laplace (Additive) smoothing. "
                                      "This is the default.")
    smoothing_types.add_argument("-gts", "--turing", action="store_true",
                                 help="Use Good Turing smoothing. Only "
                                      "available for n <= 2.")

    parser.add_argument("-p", "--perplexity", action="store_true",
                        help="Compute the perplexity of the test set, "
                             "trained on the training set.")
    parser.add_argument("-c", "--classify", metavar='FEATURE_NUM', nargs='?',
                        default="",
                        help="Binary classify the text based on perplexity on "
                             "the specified feature number (1 by default). "
                             "Ex: feature1, feature2 ,\u2026,text.")

    parser.add_argument("output_file", action="store", nargs='?')
    parser.add_argument("training_set", action="store")
    parser.add_argument("test_set", action="store", nargs='?')

    parser.usage = ("ngrams.py [-h] [-n N] -sent training_set\n         "
                    "        [-h] [-n N] [-sent] [-t T] [-ls | -gts] [-p] "
                    "[-c [FEATURE_NUM] output_file] training_set test_set")
    error_str = ""
    opts = parser.parse_args()
    if opts.classify.isdigit():
        opts.classify = int(opts.classify)
        if opts.classify <= 0:
            error_str += "Feature number must be >= 1.\n"
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
        if error_str:
            error_str += "                  "
        error_str += ("Must input an output file when performing binary "
                      "classification (-c).\n")
    if opts.perplexity or opts.classify:
        if not opts.test_set:
            if error_str:
                error_str += "                  "
            error_str += ("Must input a test set, if performing perplexity "
                          "based operations (-p and -c).\n")
        if opts.threshold < 1:
            if error_str:
                error_str += "                  "
            error_str += ("Threshold must be >= 1, if performing perplexity "
                          "based operations (-p and -c).\n")
    if opts.threshold < 0:
        if error_str:
            error_str += "                  "
        error_str += "Threshold must be > 0.\n"
    if opts.n <= 0:
        if error_str:
            error_str += "                  "
        error_str += "N must be >= 1.\n"
    if opts.turing and (opts.n > 2):
        if error_str:
            error_str += "                  "
        error_str += "Good turing smoothing is only available for n <= 2.\n"

    if error_str:
        parser.error(error_str[:-1])
    return opts


def finish_model(model, n, gts, word_freq_pairs, total_words):
    if gts:
        if n == 1:
            model.occurrenceToUniTuring(word_freq_pairs, total_words)
        else:
            model.occurrenceToBiTuring(word_freq_pairs, total_words)
    else:
        if n == 1:
            model.laplace_unigrams(word_freq_pairs, total_words)
        elif n == 2:
            model.laplace_bigrams(word_freq_pairs, total_words)
        else:
            model.laplace_ngrams(word_freq_pairs, total_words, n)


def main():
    opts = parse_args()
    n = opts.n
    gts = opts.turing
    train_str = test_str = 0

    model = Ngrams(opts)
    if opts.sentence:
        train_str, word_freq_pairs, total_words = model.init(n, 0)
        if n == 1:
            model.unsmoothed_unigrams(word_freq_pairs)
        elif n == 2:
            model.unsmoothed_bigrams(word_freq_pairs)
        else:
            model.unsmoothed_ngrams(word_freq_pairs, total_words, n)

        model.generateSentence(n) if n <= 2 else model.generateNgramSentence()

    if opts.perplexity:
        train_str, word_freq_pairs, total_words = model.init(n, 1, train_str)
        finish_model(model, n, gts, word_freq_pairs, total_words)
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
        zeroSet, oneSet, neg_class = model.processFile(n, 2, train_str)
        if n == 1:
            zero_freq_pairs = model.uni_count_pairs(zeroSet, n, True)
            zero_types = len(zero_freq_pairs)
            zero_words = len(zeroSet)
            one_freq_pairs = model.uni_count_pairs(oneSet, n, True)
            one_types = len(one_freq_pairs)
            one_words = len(oneSet)

            finish_model(model, n, gts, zero_freq_pairs, zero_words)
            zero_n = model.unigrams
            zero_ocm = model.uni_ocm

            finish_model(model, n, gts, one_freq_pairs, one_words)
            one_n = model.unigrams
            one_ocm = model.uni_ocm

            perplex_fun = model.uni_perplex
            zero_args = (t_arg, gts, zero_n, zero_words, zero_ocm, zero_types)
            one_args = (t_arg, gts, one_n, one_words, one_ocm, one_types)
        elif n == 2:
            zero_freq_pairs = model.bi_count_pairs(zeroSet, n, True)
            zero_types = len(zero_freq_pairs)
            zero_words = model.total_words
            one_freq_pairs = model.bi_count_pairs(oneSet, n, True)
            one_types = len(one_freq_pairs)
            one_words = model.total_words

            finish_model(model, n, gts, zero_freq_pairs, zero_words)
            zero_n = model.bigrams
            zero_ocm = model.bi_ocm

            finish_model(model, n, gts, one_freq_pairs, one_words)
            one_n = model.bigrams
            one_ocm = model.bi_ocm

            perplex_fun = model.bi_perplex
            zero_args = (t_arg, gts, zero_n, zero_words, zero_ocm, zero_types)
            one_args = (t_arg, gts, one_n, one_words, one_ocm, one_types)
        else:
            zero_freq_pairs = model.n_count_pairs(zeroSet, n, True)
            zero_types = len(zero_freq_pairs)
            zero_words = model.total_words
            one_freq_pairs = model.n_count_pairs(oneSet, n, True)
            one_types = len(one_freq_pairs)
            one_words = model.total_words

            finish_model(model, n, gts, zero_freq_pairs, zero_words)
            zero_n = model.ngrams

            finish_model(model, n, gts, one_freq_pairs, one_words)
            one_n = model.ngrams

            perplex_fun = model.n_laplace_perplex_help
            zero_args = (t_arg, n, zero_n, zero_words, zero_types)
            one_args = (t_arg, n, one_n, one_words, one_types)

        range_wrap = xrange if model.python2 else range
        predictions = []
        test_t = model.processFile(n, 3, test_str)
        for line in test_t:
            last_comma = -1
            for _ in range_wrap(model.num_features):
                last_comma = line.find(',', last_comma+1)
                if last_comma == -1:
                    sys.exit("More features are defined than are present.")

            t_arg.set_datum(line[last_comma+2:].split())

            # Compare perplexities
            zero_plex = perplex_fun(*zero_args)
            one_plex = perplex_fun(*one_args)

            guess = neg_class if zero_plex <= one_plex else '1'
            predictions.append(guess)

        with open(opts.output_file, 'w') as guesses:
            guesses.write('\n'.join(predictions))

if __name__ == '__main__':
    main()
