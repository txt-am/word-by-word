# coding: utf-8

from nltk.tokenize import word_tokenize
import numpy as np
import sys

PUNCTUATION = [',', '.', ';', '`', '"', "'", ":"]
INNER_PUNCTUATION = ['`', '"', "'"]


def read_vec(path):
    w2v = {}
    with open(path, 'r', encoding='utf8') as vec_file:
        first = True
        for line in vec_file:
            if first:
                first = False
                continue
            elems = line.split(' ')
            try:
                w2v[elems[0]] = np.array([float(x) for x in elems[1:]])
            except ValueError:
                print(elems)
    return w2v


def process(token):
    if token == "'m":
        return 'am'
    elif token == "n't":
        return 'not'
    elif len(token) == 1:
        return token
    else:
        for p in INNER_PUNCTUATION:
            token = token.replace(p, '')
        return token.lower()


def process_tokens(tokens):
    processed = []
    for token in tokens:
        processed.append(process(token))
    return processed


def calc_diff(vec1, vec2):
    return np.dot(vec1 - vec2, vec1 - vec2)


def get_most_similar(vec, w2v):
    best = np.inf
    most_similar = ''
    for word in w2v:
        distance = calc_diff(vec, w2v[word])
        if distance < best:
            best = distance
            most_similar = word
    return most_similar


def translate_word(word, en_vecs, de_vecs):
    if word in PUNCTUATION:
        return word
    else:
        vec = en_vecs.get(word, en_vecs.get(word.lower(), None))
        if vec is None:
            return word
        else:
            return get_most_similar(vec, de_vecs)


def translate(en_text, en_vecs, de_vecs):
    tokens = word_tokenize(en_text)
    translation = []
    for t in process_tokens(tokens):
        translation.append(translate_word(t, en_vecs, de_vecs))
    return translation


def get_sents(path):
    with open(path, 'r', encoding='utf8') as src:
        for line in src:
            yield line

            
def write_sent(path, sent):
    with open(path, 'a', encoding='utf8') as dest:
        dest.write(sent)


def translate_text_file(input_file, output_file):
    en_vecs = read_vec('wiki.multi.en.vec')
    de_vecs = read_vec('wiki.multi.de.vec')
    with open(output_file, 'w', encoding='utf8') as dest:
        for sent in get_sents(input_file):
            dest.write(' '.join(translate(sent, en_vecs, de_vecs)) + '\n')

input_file = '/data/input.txt'
output_file = '/output/output.txt'
translate_text_file(input_file, output_file)

