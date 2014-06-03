from __future__ import division, unicode_literals
import copy
import sys
import string
from math import log10
from re import sub as re_sub
from random import uniform as rand_uniform
from collections import Counter, OrderedDict, defaultdict, deque


class ngrams:
    def __init__(self, n):
        self.threshold = 1  # freq of words considered unknown
        self.start_token = "<s>"
        self.end_token = "</s>"
        self.unk_token = "<u>"
        self.unigrams = None
        self.uni_ocm = None
        self.bigrams = None
        self.bi_ocm = None
        self.ngrams = None
        self.total_words = None
        self.train_len = None
        self.types = None
        self.n = n

    def init(self, op, ts, tokens):
        self.bigrams = None
        self.bi_ocm = None
        self.total_words = None

        if not tokens:
            tokens = self.processFile(op, 0)

        unk = "-sent" not in sys.argv
        if op == 0:
            word_freq_pairs = self.uni_count_pairs(tokens, self.n, unk)
        elif op == 1:
            word_freq_pairs = self.bi_count_pairs(tokens, self.n, unk)
        else:
            word_freq_pairs = self.n_count_pairs(tokens, self.n, unk)
    
        self.types = len(word_freq_pairs)
        return tokens, word_freq_pairs

    def processFile(self, op, typ):
        tokens = ""
        f_num = 1 if typ % 2 else 2
        if sys.version_info < (3,):
            punctuation = string.punctuation.replace("?", "").replace("'", "")
            punctuation = punctuation.replace("!", "").replace(".", "")
            with open(sys.argv[len(sys.argv)-f_num], 'r') as reviews:
                tokens = unicode(reviews.read(), errors='replace')

        else:
            punctuation = string.punctuation.translate(str.maketrans(
                                                       "", "", ".?!'"))
            with open(sys.argv[len(sys.argv)-f_num], 'r',
                      errors="replace") as reviews:
                tokens = reviews.read()

        if self.n == 1:
            tokens = tokens.lower()

        # Ensure these tokens aren't in the text
        while self.start_token in tokens:
            self.start_token += '>'
        while self.end_token in tokens:
            self.end_token += '>'
        while self.unk_token in tokens:
            self.unk_token += '>'

        begin_tokens = ""
        start_tokens = ' ' + self.end_token
        for i in range(self.n-1):
            begin_tokens += ' ' + self.start_token + ' '
        start_tokens += begin_tokens

        tokens = re_sub(":\)", ' ' + "\u1F601" + ' ', tokens)
        for ch in punctuation:
            tokens = tokens.replace(ch, ' ' + ch + ' ')
        tokens = re_sub("\.\.+", ' ' + "\u2026" + ' ', tokens)
        tokens = re_sub("(\!+\?|\?+\!)[?!]*",
                        ' ' + "\u203D" + start_tokens, tokens)
        tokens = re_sub("\!\!+", " !!" + start_tokens, tokens)
        tokens = re_sub("\?\?+", " ??" + start_tokens, tokens)
        tokens = re_sub("((?<![.?!\s])[.?!])",
                        r" \1" + start_tokens, tokens)
        tokens = re_sub("(?<=[a-zI])('[a-z][a-z]?)\s", r" \1 ", tokens)
        if self.end_token not in tokens:
            tokens += start_tokens if self.n > 1 else self.end_token
        tokens = tokens.strip()

        if typ < 2:
            tmp = list(filter(bool, tokens.split()))
            tokens = []
            for i in range(self.n-1):
                tokens.append(tmp.pop())
            tokens.extend(tmp)

            if not typ:
                self.train_len = len(tokens)
            return tokens

        elif typ == 2:
            zeroSet = []
            oneSet = []
            tokens = tokens[-len(begin_tokens):] + tokens[:-len(begin_tokens)]
            tokens = tokens.split('\n')
            del tokens[0]
            for line in tokens:
                clas = line[0]
                line = line[8:]
                if clas != '1':  # sometimes 0 and sometimes -1
                    zeroSet.extend(line.split())
                else:
                    oneSet.extend(line.split())

            for line in tokens:
                if line[0] != 1:
                    return zeroSet, oneSet, line[0]
            sys.exit("Only one class in the training set. Exiting now.")

        return tokens[-len(begin_tokens):] + tokens[:-len(begin_tokens)]

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
            for word, count in word_freq_pairs.items():
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
            for word, count in self.total_words.items():
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
            for nxt_lvl_dict in word_freq_pairs.values():
                for word, count in list(nxt_lvl_dict.items()):
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
            for word, count in self.total_words.items():
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
        for word, count in prob_dict.items():
            prob_dict[word] = count / self.total_words

        self.unigrams = prob_dict

    def unsmoothed_bigrams(self, word_freq_pairs):
        prob_dict = word_freq_pairs
        for word, nxt_lvl_dict in prob_dict.items():
            for word_infront, cnt in nxt_lvl_dict.items():
                nxt_lvl_dict[word_infront] = cnt / self.total_words[word]

        self.bigrams = prob_dict

    def unsmoothed_ngrams(self, word_freq_pairs, total_words, n):
        prob_dict = word_freq_pairs
        if n == 2:
            for word, nxt_lvl_dict in prob_dict.items():
                for word_infront, count in nxt_lvl_dict.items():
                    nxt_lvl_dict[word_infront] = count / total_words[word]
            return

        for word in prob_dict:
            self.unsmoothed_ngrams(prob_dict[word], total_words[word], n - 1)

        self.ngrams = prob_dict

    """
    Computes Laplace smoothed probability distributions.
    """
    def laplace_unigrams(self, word_freq_pairs, total_words=None):
        if not total_words:
            total_words = self.total_words

        prob_dict = word_freq_pairs
        dict_len = len(prob_dict)
        for word, count in prob_dict.items():
            prob_dict[word] = (count+1) / (total_words+dict_len)

        self.unigrams = prob_dict

    def laplace_bigrams(self, word_freq_pairs, total_words=None):
        if not total_words:
            total_words = self.total_words

        prob_dict = word_freq_pairs
        dict_len = len(prob_dict)
        for top_word, nxt_lvl_dict in prob_dict.items():
            for bot_word, cnt in nxt_lvl_dict.items():
                nxt_lvl_dict[bot_word] = ((cnt+1) /
                                          (total_words[top_word] +  dict_len))
        self.bigrams = prob_dict

    def laplace_ngrams(self, word_freq_pairs, total_words, n):
        prob_dict = word_freq_pairs
        if n == 2:
            dict_len = len(prob_dict)
            for top_word, nxt_lvl_dict in prob_dict.items():
                for bot_word, cnt in nxt_lvl_dict.items():
                    nxt_lvl_dict[bot_word] = ((cnt+1) /
                                              (total_words[top_word] +
                                               dict_len))
            return

        for word in prob_dict:
            self.laplace_ngrams(prob_dict[word], total_words[word], n - 1)

        self.ngrams = prob_dict

    """
    Creates a dict of how many times a word of a certain frequency occurs.
    Then gets probabilty distributions from good turing smoothing.
    """
    def occurrenceToUniTuring(self, word_freq_pairs, total_words=None):
        occurence_map = OrderedDict.fromkeys(range(1, max(
                                             word_freq_pairs.values())+2), 0)

        for value in word_freq_pairs.values():
            occurence_map[value] += 1
        if word_freq_pairs[self.unk_token] <= self.threshold:
            occurence_map[word_freq_pairs[self.unk_token]] = 1

        #fill in the levels with 0 words
        last_val = 1
        for key, value in reversed(list(occurence_map.items())):
            if not value:
                occurence_map[key] = last_val
            last_val = occurence_map[key]
        self.uni_ocm = occurence_map

        if not total_words:
            total_words = self.total_words
        self.goodTuringSmoothUni(word_freq_pairs, occurence_map, total_words)

    def goodTuringSmoothUni(self, word_freq_pairs, uni_ocm, total_words):
        prob_dict = word_freq_pairs
        for word, count in prob_dict.items():
            prob_dict[word] = ((count+1) * uni_ocm[count+1] / uni_ocm[count] /
                              total_words)

        self.unigrams = prob_dict

    def occurrenceToBiTuring(self, word_freq_pairs, total_words=None):
        unk_token = self.unk_token
        occurence_map = dict.fromkeys(word_freq_pairs.keys())
        for word, nxt_lvl_dict in word_freq_pairs.items():
            if nxt_lvl_dict:
                unk_words = set()
                for second_word, count in nxt_lvl_dict.items():
                    if count <= self.threshold:
                        unk_words.add(second_word)

                unk_words.discard(self.start_token)
                unk_words.discard(self.end_token)
                unk_words.discard(unk_token)
                nxt_lvl_dict.update((unk_token, nxt_lvl_dict[unk_token] + cnt)
                                    for wrd2, cnt in list(nxt_lvl_dict.items())
                                    if wrd2 in unk_words)

                for unk_word in unk_words:
                    del nxt_lvl_dict[unk_word]

                top = max(nxt_lvl_dict.values())
                occurence_map[word] = OrderedDict.fromkeys(range(1, top+2), 0)
                for value in nxt_lvl_dict.values():
                    occurence_map[word][value] += 1
            else:
                # "Fill" with unk_token, if empty
                occurence_map[word] = {1: 1}
                self.total_words[word] = 1

            #fill in the levels with 0 words
            last_val = 1
            for key, value in reversed(list(occurence_map[word].items())):
                if not value:
                    occurence_map[word][key] = last_val
                last_val = occurence_map[word][key]

        self.bi_ocm = occurence_map
        if not total_words:
            total_words = self.total_words
        self.goodTuringSmoothBi(word_freq_pairs, total_words, occurence_map)

    def goodTuringSmoothBi(self, word_freq_pairs, total_words, bi_ocm):
        prob_dict = word_freq_pairs
        for w, infront_dict in prob_dict.items():
            for w_infront, cnt in infront_dict.items():
                infront_dict[w_infront] = ((cnt+1) * bi_ocm[w][cnt+1] /
                                           bi_ocm[w][cnt] / total_words[w])
        self.bigrams = prob_dict

    """
    Generates sentences based on probability distributions.
    """
    def generateSentence(self, op):
        word = self.start_token
        sentence = []

        word = self.weightedPickBi(word) if op else self.weightedPickUni()
        while word != self.end_token:
            sentence.append(word)
            word = self.weightedPickBi(word) if op else self.weightedPickUni()

        print(' '.join(sentence))

    def weightedPickUni(self):
        s = 0.0
        key = ""
        r = rand_uniform(0, sum(self.unigrams.values()))
        for key, weight in self.unigrams.items():
            s += weight
            if r < s:
                return key
        return key

    def weightedPickBi(self, word):
        s = 0.0
        key = ""
        r = rand_uniform(0, sum(self.bigrams[word].values()))
        for key, weight in self.bigrams[word].items():
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
        r = rand_uniform(0, sum(tmp_dict.values()))
        for key, weight in tmp_dict.items():
            s += weight
            if r < s:
                return key
        return key

    def uni_perplex(self, tokens, ts, unigrams=None,
                    train_len=None, uni_ocm=None, V=None):
        if not unigrams:
            unigrams = self.unigrams
        if not train_len:
            train_len = self.train_len

        entropy = 0.0
        if ts:
            if not uni_ocm:
                uni_ocm = self.uni_ocm
            for token in tokens:
                entropy -= log10(unigrams.get(token,
                                 uni_ocm[self.threshold] / train_len))
        else:
            if not V:
                V = self.types
            for token in tokens:
                entropy -= log10(unigrams.get(token, 1 / (train_len + V)))

        return 10**(entropy / (len(tokens) - (self.n-1)))

    def bi_perplex(self, tokens, ts, bigrams=None,
                   tw=None, bi_ocm=None, V=None):
        ut = self.unk_token
        thresh = self.threshold
        if not tw:
            tw = self.total_words
        if not bigrams:
            bigrams = self.bigrams

        entropy = 0.0
        prev_t = tokens[0]
        if ts:
            if not bi_ocm:
                bi_ocm = self.bi_ocm
            for token in tokens[1:]:
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
            for token in tokens[1:]:
                if prev_t in bigrams:
                    entropy -= log10(bigrams[prev_t].get(token,
                                     1 / (tw[prev_t] + V)))
                else:
                    entropy -= log10(bigrams[ut].get(token, 1 / (tw[ut] + V)))

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

    def n_laplace_perplex(self, tokens, ngrams, total_words, n):
        help_dict = ngrams
        if n == 1:
            return log10(help_dict.get(tokens[0], 1 /
                                       (total_words + self.types)))
        nxt_token = tokens.popleft()
        if nxt_token in help_dict:
            return self.n_laplace_perplex(tokens, help_dict[nxt_token],
                                          total_words[nxt_token], n - 1)
        """
        if hash(''.join(help_dict.keys())) in sum_dict:
            total = sum_dict[hash(''.join(help_dict.keys()))]
        else:
            total = self.sum_count(help_dict.values())
            sum_dict[hash(''.join(help_dict.keys()))] = total
        """
        # small "punishment" for not being in it (100), real approx is too slow
        return log10(1 / (100 + self.types))

    def n_laplace_perplex_help(self, tokens, n):
        #sum_dict = {}
        entropy = 0.0
        num_tokens = len(tokens)
        words = deque(tokens[:n])
        for i in range(num_tokens - n):
            entropy -= self.n_laplace_perplex(copy.copy(words), self.ngrams,
                                              self.total_words, n)
            del words[0]
            words.append(tokens[i+n])
        entropy -= self.n_laplace_perplex(words, self.ngrams,
                                          self.total_words, n)

        return 10**(entropy / (num_tokens - (n-1)))


def finish_model(model, n, ts, word_freq_pairs, total_words):
    if ts:
        if n == 1:
            model.occurrenceToUniTuring(word_freq_pairs, total_words)
        else:
            model.occurrenceToBiTuring(word_freq_pairs, total_words)
    else:
        if n == 1:
            model.laplace_unigrams(word_freq_pairs)
        elif n == 2:
            model.laplace_bigrams(word_freq_pairs)
        else:
            model.laplace_ngrams(word_freq_pairs, total_words, n)


def main():
    # Flag and argument upkeep
    op = 0
    n = "-n" in sys.argv
    ls = "-ls" in sys.argv
    perplex = '-p' in sys.argv
    classify = "--classify" in sys.argv
    if n:
        n = int(sys.argv[sys.argv.index("-n") + 1])
        if n == 1:
            op = 0
            ts = "-ts" in sys.argv
        elif n == 2:
            op = 1
            ts = "-ts" in sys.argv
        else:
            op = 2
            ts = False
    else:
        n = 2
        ts = "-ts" in sys.argv
        op = 0 if "-u" in sys.argv else 1

    model = ngrams(n)
    tokens, word_freq_pairs = model.init(op, ts, None)

    if "-sent" in sys.argv:
        if op == 0:
            model.unsmoothed_unigrams(word_freq_pairs)
        elif op == 1:
            model.unsmoothed_bigrams(word_freq_pairs)
        else:
            model.unsmoothed_ngrams(word_freq_pairs, model.total_words, n)

        model.generateSentence(op) if op < 2 else model.generateNgramSentence()
        if not (perplex or classify):
            sys.exit(0)

    # only want perplexity
    elif perplex and not classify:
        finish_model(model, n, ts, word_freq_pairs, model.total_words)
        tokens = model.processFile(op, 1)
        if n == 1:
            perplexity = model.uni_perplex(tokens, ts)
        elif n == 2:
            perplexity = model.bi_perplex(tokens, ts)
        else:
            perplexity = model.n_laplace_perplex_help(tokens, n)
        print("Perplexity: " + str(perplexity))
        sys.exit(0)

    if perplex:
        if op == 0:
            word_freq_pairs = model.uni_count_pairs(tokens, n, True)
        elif op == 1:
            word_freq_pairs = model.bi_count_pairs(tokens, n, True)
        else:
            word_freq_pairs = model.n_count_pairs(tokens, n, True)

        finish_model(model, n, ts, word_freq_pairs, model.total_words)
        tokens = model.processFile(op, 1)
        if n == 1:
            perplexity = model.uni_perplex(tokens, ts)
        elif n == 2:
            perplexity = model.bi_perplex(tokens, ts)
        else:
            perplexity = model.n_laplace_perplex_help(tokens, n)
        print("Perplexity: " + str(perplexity))

    if classify:
        zeroSet, oneSet, neg_class = model.processFile(op, 2)
        if n == 1:
            zero_freq_pairs = model.uni_count_pairs(zeroSet, n, True)
            zero_words = len(zeroSet)
            one_freq_pairs = model.uni_count_pairs(oneSet, n, True)
            one_words = len(oneSet)
            
            finish_model(model, n, ts, zero_freq_pairs, zero_words)
            zero_n = model.unigrams
            zero_ocm = model.uni_ocm

            finish_model(model, n, ts, one_freq_pairs, one_words)
            one_n = model.unigrams
            one_ocm = model.uni_ocm

            perplex_fun = model.uni_perplex
        elif n == 2:
            zero_freq_pairs = model.bi_count_pairs(zeroSet, n, True)
            zero_words = model.total_words
            one_freq_pairs = model.bi_count_pairs(oneSet, n, True)
            one_words = model.total_words

            finish_model(model, n, ts, zero_freq_pairs, zero_words)
            zero_n = model.bigrams
            zero_ocm = model.bi_ocm

            finish_model(model, n, ts, one_freq_pairs, one_words)
            one_n = model.bigrams
            one_ocm = model.bi_ocm

            perplex_fun = model.bi_perplex
        else:
            zero_freq_pairs = model.n_count_pairs(zeroSet, n, True)
            zero_words = model.total_words
            one_freq_pairs = model.n_count_pairs(oneSet, n, True)
            one_words = model.total_words
            
            finish_model(model, n, ts, zero_freq_pairs, zero_words)
            zero_n = model.ngrams

            finish_model(model, n, ts, one_freq_pairs, one_words)
            one_n = model.ngrams

            perplex_fun = model.n_laplace_perplex_help

        predictions = []
        tokens = model.processFile(op, 3)
        for line in tokens.split('\n')[1:]:
            line = line[8:].split()

            # Compare perplexities
            zero_plex = perplex_fun(line, ts, zero_n, zero_words,
                                    zero_ocm, len(zero_n))
            one_plex = perplex_fun(line, ts, one_n, one_words,
                                   one_ocm, len(one_n))
            
            guess = neg_class if zero_plex <= one_plex else '1'
            predictions.append(guess)

        with open('kaggle_dump1.txt', 'w') as guesses:
            guesses.write('\n'.join(predictions))

if __name__ == '__main__':
    main()
