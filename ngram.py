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
        if typ == 0:
            if sys.version_info < (3,):
                punctuation = string.punctuation.replace("?", "").replace("'", "")
                punctuation = punctuation.replace("!", "").replace(".", "")
                with open(sys.argv[len(sys.argv)-2], 'r') as reviews:
                    tokens = unicode(reviews.read(), errors='replace')

            else:
                punctuation = string.punctuation.translate(str.maketrans(
                                                           "", "", ".?!'"))
                with open(sys.argv[len(sys.argv)-2], 'r',
                          errors="replace") as reviews:
                    tokens = reviews.read()

            if not op:
                tokens = tokens.lower()

            # Ensure these tokens aren't in the text
            while self.start_token in tokens:
                self.start_token += '>'
            while self.end_token in tokens:
                self.end_token += '>'
            while self.unk_token in tokens:
                self.unk_token += '>'

            start_tokens = ' ' + self.end_token
            for i in range(self.n-1):
                start_tokens += ' ' + self.start_token + ' '

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
            if not self.start_token in tokens:
                tokens += start_tokens

            tmp = list(filter(bool, tokens.strip().split()))
            tokens = []
            for i in range(self.n-1):
                tokens.append(tmp.pop())
            tokens.extend(tmp)

            self.train_len = len(tokens)
            return tokens

        elif typ == 1:
            if sys.version_info < (3,):
                punctuation = string.punctuation.replace("?", "").replace("'", "")
                punctuation = punctuation.replace("!", "").replace(".", "")
                with open(sys.argv[len(sys.argv)-1], 'r') as reviews:
                    tokens = unicode(reviews.read(), errors='replace')

            else:
                punctuation = string.punctuation.translate(str.maketrans(
                                                           "", "", ".?!'"))
                with open(sys.argv[len(sys.argv)-1], 'r',
                          errors="replace") as reviews:
                    tokens = reviews.read()

            if not op:
                tokens = tokens.lower()

            while self.start_token in tokens:
                self.start_token += '>'
            while self.end_token in tokens:
                self.end_token += '>'
            while self.unk_token in tokens:
                self.unk_token += '>'

            start_tokens = ' ' + self.end_token
            for i in range(self.n-1):
                start_tokens += ' ' + self.start_token + ' '

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
            if not self.start_token in tokens:
                tokens += start_tokens

            tmp = list(filter(bool, tokens.strip().split()))
            tokens = []
            for i in range(self.n-1):
                tokens.append(tmp.pop())
            tokens.extend(tmp)
            return tokens

        elif typ == 2:
            zeroSet = []
            oneSet = []
            with open('reviews.train', 'r') as reviews:
                for line in reviews.readlines()[1:]:
                    clas = line[0]
                    line = line[4:]
                    if clas == '0':
                        zeroSet.append(unicode(line, errors='replace'))
                    else:
                        oneSet.append(unicode(line, errors='replace'))

                if sys.version_info < (3,):
                    punctuation = string.punctuation.translate(None, ".?!'")
                else:
                    punctuation = string.punctuation.translate(str.maketrans(
                                                               "", "", ".?!'"))
                zeroSet = ' '.join(zeroSet)
                zeroSet = re_sub(":\)", ' ' + "\u1F601" + ' ', zeroSet)
                for ch in punctuation:
                    zeroSet = zeroSet.replace(ch, ' ' + ch + ' ')
                zeroSet = re_sub("\.\.+", ' ' + "\u2026" + ' ', zeroSet)
                zeroSet = re_sub("(\!+\?|\?+\!)[?!]*",
                                 ' ' + "\u203D" + start_tokens, zeroSet)
                zeroSet = re_sub("\!\!+", " !!" + start_tokens, zeroSet)
                zeroSet = re_sub("\?\?+", " ??" + start_tokens, zeroSet)
                zeroSet = re_sub("((?<![.?!\s])[.?!])",
                                 r" \1" + start_tokens, zeroSet)
                zeroSet = re_sub("(?<=[a-zI])('[a-z][a-z]?)\s",
                                 r" \1 ", zeroSet)
                zeroSet = filter(bool, zeroSet.strip().split())
                zeroSet.insert(0, zeroSet.pop())

                oneSet = ' '.join(oneSet)
                oneSet = re_sub(":\)", ' ' + "\u1F601" + ' ', oneSet)
                for ch in punctuation:
                    oneSet = oneSet.replace(ch, ' ' + ch + ' ')
                oneSet = re_sub("\.\.+", ' ' + "\u2026" + ' ', oneSet)
                oneSet = re_sub("(\!+\?|\?+\!)[?!]*",
                                ' ' + "\u203D" + start_tokens, oneSet)
                oneSet = re_sub("\!\!+", " !!" + start_tokens, oneSet)
                oneSet = re_sub("\?\?+", " ??" + start_tokens, oneSet)
                oneSet = re_sub("((?<![.?!\s])[.?!])",
                                r" \1" + start_tokens, oneSet)
                oneSet = re_sub("(?<=[a-zI])('[a-z][a-z]?)\s", r" \1 ", oneSet)
                oneSet = filter(bool, oneSet.strip().split())
                oneSet.insert(0, oneSet.pop())

                self.set1_len = len(oneSet)
                self.set0_len = len(zeroSet)
                return oneSet, zeroSet

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
            self.total_words[self.unk_token] = 0.0

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
    def laplace_unigrams(self, word_freq_pairs):
        prob_dict = word_freq_pairs
        dict_len = len(prob_dict)
        for word, count in prob_dict.items():
            prob_dict[word] = (count+1) / (self.total_words+dict_len)

        self.unigrams = prob_dict

    def laplace_bigrams(self, word_freq_pairs):
        prob_dict = word_freq_pairs
        dict_len = len(prob_dict)
        for top_word, nxt_lvl_dict in prob_dict.items():
            for bot_word, cnt in nxt_lvl_dict.items():
                nxt_lvl_dict[bot_word] = ((cnt+1) /
                                          (self.total_words[top_word] +
                                           dict_len))
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
    def occurrenceToUniTuring(self, word_freq_pairs):
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
        self.goodTuringSmoothUni(word_freq_pairs)

    def goodTuringSmoothUni(self, word_freq_pairs):
        prob_dict = word_freq_pairs
        for word, count in prob_dict.items():
            prob_dict[word] = ((count+1) * self.uni_ocm[count+1] /
                               self.uni_ocm[count]) / self.total_words

        self.unigrams = prob_dict

    def occurrenceToBiTuring(self, word_freq_pairs):
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
        self.goodTuringSmoothBi(word_freq_pairs)

    def goodTuringSmoothBi(self, word_freq_pairs):
        prob_dict = word_freq_pairs
        for w, infront_dict in prob_dict.items():
            for w_infront, cnt in infront_dict.items():
                infront_dict[w_infront] = (((cnt+1) * self.bi_ocm[w][cnt+1] /
                                           self.bi_ocm[w][cnt]) /
                                           self.total_words[w])
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

    def uni_perplex(self, tokens, ts):
        entropy = 0.0
        if ts:
            for token in tokens:
                entropy -= log10(self.unigrams.get(token,
                                 self.uni_ocm[self.threshold] /
                                 self.train_len))
        else:
            for token in tokens:
                entropy -= log10(self.unigrams.get(token, 1 / self.train_len))

        return 10**(entropy / (len(tokens) - (self.n-1)))

    def bi_perplex(self, tokens, ts):
        bigrams = self.bigrams
        thresh = self.threshold
        tw = self.total_words
        bi_ocm = self.bi_ocm
        ut = self.unk_token
        V = self.types

        entropy = 0.0
        prev_t = tokens[0]
        if ts:
            for token in tokens[1:]:
                if prev_t in bigrams:
                    entropy -= log10(bigrams[prev_t].get(token,
                                     bi_ocm[prev_t][thresh] / tw[prev_t]))
                else:
                    entropy -= log10(bigrams[ut].get(token,
                                     bi_ocm[ut][thresh] / tw[ut]))
                prev_t = token
        else:
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

    """
    def bi_comp(self, tokens, bigrams, total_words, ocm_bi):
        perplexity = 0.0
        prev_token = tokens[0]

        for token in tokens[1:]:
            if prev_token in bigrams:
               perplexity += numpy.log(bigrams[prev_token].get(token, ocm_bi[prev_token][self.threshold] /
                                                                total_words[prev_token]))
            else:
                perplexity += numpy.log(bigrams[self.unk_token].get(token, ocm_bi[self.unk_token][self.threshold] /
                                        total_words[self.unk_token]))
            prev_token = token

        return numpy.power(1 / numpy.exp(perplexity), 1.0 / len(tokens))
    """


def main():
    # Flag and argument upkeep
    op = 0
    n = "-n" in sys.argv
    if n:
        n = int(sys.argv[sys.argv.index("-n") + 1])
        if n == 1:
            op = 0
            ts = "-ts" in sys.argv
            ls = "-ls" in sys.argv
        elif n == 2:
            op = 1
            ts = "-ts" in sys.argv
            ls = "-ls" in sys.argv
        else:
            op = 2
            ls = "-ls" in sys.argv
            ts = False
    else:
        op = 0 if "-u" in sys.argv else 1
        n = 2
        ts = "-ts" in sys.argv
        ls = "-ls" in sys.argv

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
        if not (ts or ls):
            sys.exit(0)

    # only want perplexity
    else:
        if ts:
            if op:
                model.occurrenceToBiTuring(word_freq_pairs)
            else:
                model.occurrenceToUniTuring(word_freq_pairs)
        else:
            if n == 1:
                model.laplace_unigrams(word_freq_pairs)
            elif n == 2:
                model.laplace_bigrams(word_freq_pairs)
            else:
                model.laplace_ngrams(word_freq_pairs, model.total_words, n)

        tokens = model.processFile(op, 1)
        if n == 1:
            perplexity = model.uni_perplex(tokens, ts)
        elif n == 2:
            perplexity = model.bi_perplex(tokens, ts)
        else:
            perplexity = model.n_laplace_perplex_help(tokens, n)
        print("Perplexity: " + str(perplexity))
        sys.exit(0)

    # want both
    if op == 0:
        word_freq_pairs = model.uni_count_pairs(tokens, n, True)
    elif op == 1:
        word_freq_pairs = model.bi_count_pairs(tokens, n, True)
    else:
        word_freq_pairs = model.n_count_pairs(tokens, n, True)

    if ts:
        if op:
            model.occurrenceToBiTuring(word_freq_pairs)
        else:
            model.occurrenceToUniTuring(word_freq_pairs)
    else:
        if n == 1:
            model.laplace_unigrams(word_freq_pairs)
        elif n == 2:
            model.laplace_bigrams(word_freq_pairs)
        else:
            model.laplace_ngrams(word_freq_pairs, model.total_words, n)

    tokens = model.processFile(op, 1)
    if n == 1:
        perplexity = model.uni_perplex(tokens, ts)
    elif n == 2:
        perplexity = model.bi_perplex(tokens, ts)
    else:
        perplexity = model.n_laplace_perplex_help(tokens, n)
    print("Perplexity: " + str(perplexity))

    """
    (oneSet, zeroSet) = model.processFile(op, 2)

    (one_bi, one_words, one_ocm) = model.init(op, ts, oneSet)

    zero_bi, zero_words, zero_ocm) = model.init(op, ts, zeroSet)


    predictions = []
    with open(sys.argv[len(sys.argv)-1]+'.test', 'r') as reviews:
        for line in reviews.readlines()[1:]:
            line = line[2:]

            line = re.sub(":\)", ' ' + u"\u1F601" + ' ', line) #smileys
            line = re.sub("\)", " ) ", line)
            line = re.sub("\(", " ( ", line)
            line = re.sub(":", " : ", line)
            line = re.sub(";", " ; ", line)
            line = re.sub("\.\.+", ' ' + u"\u2026" + ' ', line) #elipsis
            line = re.sub("\."," . </s> <s> ", line)
            line = re.sub("([!?]{2,10})+", ' ' + u"\u203D" + " </s> <s> ", line) #interrobang
            line = re.sub("!!+"," !! </s> <s> ", line)
            line = re.sub("!"," ! </s> <s> ", line)
            line = re.sub("\?\?+"," ?? </s> <s> ", line)
            line = re.sub("\?"," ? </s> <s> ", line).split()
            line = filter(bool, line)
            line.insert(0, line.pop())

            #compare perplexities
            one_plex = model.bi_comp(line, one_bi, one_words, one_ocm)
            zero_plex = model.bi_comp(line, zero_bi, zero_words, zero_ocm)

            if zero_plex <= one_plex:
                predictions.append(0)
            else:
                predictions.append(1)

    with open('kaggle_dump.txt', 'w') as guesses:
        for i,guess in enumerate(predictions):
            guesses.write('%d,%i\n' % (i, guess))

    """

if __name__ == '__main__':
    main()
