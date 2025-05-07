import os
import torch
import pickle
import random

from collections import Counter
from sklearn.model_selection import ShuffleSplit


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0
        self.freeze = False

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, use_unk, data_percentage, seed, n_folds, fold):
        self.seed = seed
        self.dictionary = Dictionary()
        self.valid_words = 0
        self.valid_chars = 0
        if use_unk:
            if n_folds > 0:
                lines = []
                with open(os.path.join(path, 'train.txt')) as f:
                    lines += f.readlines()
                with open(os.path.join(path, 'valid.txt')) as f:
                    lines += f.readlines()
                ss = ShuffleSplit(n_splits=10, test_size=0.25, random_state=seed)
                i = 0

                train_lines = []
                test_lines = []
                for train_index, test_index in ss.split(lines):
                    if i == fold:
                        train_lines = [lines[i] for i in train_index]
                        test_lines = [lines[i] for i in test_index]
                        break
                    i += 1

                for line in test_lines:
                    line = line.replace("\n", "").replace("## ", "")
                    self.valid_words += len(line.split()) + 1
                    self.valid_chars += len(line.replace("$$", "").replace(" ", "")) + 1

                with open(os.path.join(path, 'temp_train.txt'), 'w') as f:
                    f.writelines(train_lines)
                with open(os.path.join(path, 'temp_valid.txt'), 'w') as f:
                    f.writelines(test_lines)
                self.train = self.tokenize(os.path.join(path, 'temp_train.txt'), False, data_percentage)
                self.valid = self.tokenize(os.path.join(path, 'temp_valid.txt'))
                os.remove(os.path.join(path, 'temp_train.txt'))
                os.remove(os.path.join(path, 'temp_valid.txt'))
            else:
                self.train = self.tokenize(os.path.join(path, 'train.txt'), False, data_percentage)
                self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
                self.test = self.tokenize(os.path.join(path, 'test.txt'))
        else:
            if n_folds > 0:
                lines = []
                with open(os.path.join(path, 'train.txt')) as f:
                    lines += f.readlines()
                with open(os.path.join(path, 'valid.txt')) as f:
                    lines += f.readlines()
                ss = ShuffleSplit(n_splits=10, test_size=0.25, random_state=seed)
                i = 0

                train_lines = []
                test_lines = []
                for train_index, test_index in ss.split(lines):
                    if i == fold:
                        train_lines = [lines[i] for i in train_index]
                        test_lines = [lines[i] for i in test_index]
                        break
                    i += 1

                with open(os.path.join(path, 'temp_train.txt'), 'w') as f:
                    f.writelines(train_lines)
                with open(os.path.join(path, 'temp_valid.txt'), 'w') as f:
                    f.writelines(test_lines)
                self.train = self.tokenize(os.path.join(path, 'temp_train.txt'), False, data_percentage)
                self.valid = self.tokenize(os.path.join(path, 'temp_valid.txt'), False)
                os.remove(os.path.join(path, 'temp_train.txt'))
                os.remove(os.path.join(path, 'temp_valid.txt'))
            else:
                self.train = self.tokenize(os.path.join(path, 'train.txt'), False, data_percentage)
                self.valid = self.tokenize(os.path.join(path, 'valid.txt'), False)
                self.test = self.tokenize(os.path.join(path, 'test.txt'), False)

    def tokenize(self, path, use_unk=True, data_percentage=100):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        unk_words = set()
        with open(path, 'r') as f:
            lines = f.readlines()

        random.Random(self.seed).shuffle(lines)
        end = len(lines) if data_percentage == 100 else int(float(len(lines))/100*data_percentage)
        lines = lines[:end]

        self.dictionary.add_word('<unk>')
        
        tokens = 0
        for line in lines:
            words = line.split() + ['<eos>']
            tokens += len(words)
            for word in words:
                if use_unk:
                    if word in self.dictionary.word2idx:
                        unk_words.discard(word)
                        continue
                    else:
                        self.dictionary.add_word(word)
                        unk_words.add(word)
                        continue
                else:
                    self.dictionary.add_word(word)
        
        # Tokenize file content
        ids = torch.LongTensor(tokens)
        token = 0
        for line in lines:
            words = line.split() + ['<eos>']
            for word in words:
                if word in unk_words:
                    word = '<unk>'
                ids[token] = self.dictionary.word2idx[word]
                token += 1

        return ids
