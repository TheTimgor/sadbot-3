import json
import os
import pickle
import numpy
import math
import nltk
from nltk import NaiveBayesClassifier
from nltk import word_tokenize
from nltk.parse import stanford
from nltk.tag import StanfordNERTagger
from nltk.corpus import nps_chat
from sacremoses import MosesDetokenizer
from heapq import nlargest
from collections import Counter
from random import choice, sample
# fuck pycharm
# noinspection PyUnresolvedReferences
from pattern.en import conjugate, tenses

cwd = os.getcwd()
os.environ['CLASSPATH'] = os.getcwd() + '/nlp/stanford-parser-full-2018-10-17'

with open('config.json') as f:
    config = json.load(f)

history = []
vocab = {}
named_entities = []
detokenizer = MosesDetokenizer()
parser = stanford.StanfordParser(model_path=cwd+'/nlp/englishPCFG.ser.gz')
tagger = StanfordNERTagger(cwd+'/nlp/stanford-ner-2018-10-16/classifiers/english.muc.7class.distsim.crf.ser.gz',
                           cwd+'/nlp/stanford-ner-2018-10-16/stanford-ner-3.9.2.jar')


def traverse_for(tree, tags):
    # print(tree)
    if type(tree) == str:
        return None
    elif tree.label() in tags:
        return tree.leaves()
    elif tree[0]:
        for a in tree:
            trav = traverse_for(a, tags)
            if trav is not None:
                return trav


def sentence_features(s, v=vocab):
    if type(s) == str:
        s = word_tokenize(s)
    s_words = set(s)
    features = {}
    for w in v:
        features[f'contains({w})'] = w in s_words
    return features


def generate_greeting_classifier(s):
    train, test = s[100:], s[:100]
    global greeting_classifier
    greeting_classifier = NaiveBayesClassifier.train(train)
    print(nltk.classify.accuracy(greeting_classifier, test))


def generate_greeting_classifier_nps():
    global greeting_classifier
    try:
        with open('greet_classifier.pickle', 'rb') as f:
            greeting_classifier = pickle.load(f)
    except FileNotFoundError:
        v = set([w.lower() for w in nps_chat.words()])
        posts = nps_chat.xml_posts()[:5000]
        h = [(sentence_features(s.text.lower(), v=v), s.get('class') if s.get('class') in ['Greet', 'Bye'] else 'Other')
             for s in posts]
        generate_greeting_classifier(h)
        with open('greet_classifier.pickle', 'wb') as f:
            pickle.dump(greeting_classifier, f)


def classify_greeting(s):
    v = set([w.lower() for w in nps_chat.words()])
    return greeting_classifier.classify(sentence_features(s.lower(), v=v))


def classify_question(s):
    if type(s) == str:
        s = parser.parse_one(word_tokenize(s))
    if traverse_for(s, ['SBARQ']):
        return 'wh-question'
    elif traverse_for(s, ['SQ']):
        return 'y/n-question'
    else:
        return 'other'


def cosine_dic(dic1, dic2):
    numerator = 0
    dena = 0
    for key1 in dic1:
        val1 = dic1[key1]
        numerator += val1*dic2.get(key1, 0.0)
        dena += val1*val1
    denb = 0
    for val2 in dic2.values():
        denb += val2*val2
    try:
        return numerator/math.sqrt(dena*denb)
    except ZeroDivisionError:
        return 0


def word_vectorize(sent):
    vector = {}
    words = word_tokenize(sent)
    counts = dict(Counter(words))
    for w in vocab:
        if w in counts:
            vector[w] = counts[w] / vocab[w]
        else:
            vector[w] = 0
    # print({a: vector[a] for a in vector if vector[a] > 0})
    return vector


def find_question_root(s):
    if type(s) == str:
        s = word_tokenize(s)
    t = parser.parse_one(s)
    # t.draw()
    vp = traverse_for(t, ['VBP', 'VBD', 'VBZ', 'MD'])
    np = traverse_for(t, ['NP'])
    return np, vp


def fix_np(np):
    np = [a.lower() for a in np]
    if 'you' in np and ('i' in np or 'me' in np):
        return np
    if 'you' in np:
        return ['i' if a == 'you' else a for a in np]
    if 'i' in np:
        return ['you' if a == 'i' else a for a in np]
    return np


def fix_vp(np, vp):
    verb = detokenizer.detokenize(vp)
    tnss = tenses(verb)
    if np == ['i']:
        tns = [a for a in tnss if 2 in a][0]
        return [conjugate(verb,
                          tense=tns[0],
                          person=1,
                          number=tns[2],
                          mood=tns[3],
                          aspect=tns[4])]
    if np == ['you']:
        tns = [a for a in tnss if 1 in a][0]
        return [conjugate(verb,
                          tense=tns[0],
                          person=2,
                          number=tns[2],
                          mood=tns[3],
                          aspect=tns[4])]
    return vp


def uninvert(s):
    np, vp = find_question_root(s)
    np = fix_np(np)
    vp = fix_vp(np, vp)
    print(np,vp)
    return detokenizer.detokenize(np)+' '+detokenizer.detokenize(vp)


def why_answer(s):
    np, vp = find_question_root(s)
    np = fix_np(np)
    return 'because '+detokenizer.detokenize(np)


def build_model(h):
    global model
    model = Model(h)
    global history
    history = h
    global vocab
    for w in word_tokenize('\n'.join(h)):
        if w.lower() in vocab:
            vocab[w.lower()] += 1
        else:
            vocab[w.lower()] = 1
    # print(vocab)
    global named_entities
    try:
        with open('named.pickle', 'rb') as f:
            named_entities = pickle.load(f)
    except FileNotFoundError:
        h_tokens = [word_tokenize(s) for s in h]
        tagged = tagger.tag_sents(h_tokens)
        named_entities = [tagged[0][0]]
        for n_e in tagged:
            for i in range(1, len(n_e)):
                if n_e[i][1] == n_e[i - 1][1]:
                    named_entities[-1] = (named_entities[-1][0] + ' ' + n_e[i][0], n_e[i][1])
                else:
                    named_entities.append(n_e[i])
        print(named_entities)
        with open('named.pickle', 'wb') as f:
            pickle.dump(named_entities, f)
    generate_greeting_classifier_nps()
    # print('finding greetings')
    # greeting_classified = {s: classify_greeting(s) for s in h[:100]}
    # print('found greetings')
    global hellos, byes
    # hellos = {s: greeting_classified[s] for s in greeting_classified if greeting_classified[s] == 'Greet'}
    # byes = {s: greeting_classified[s] for s in greeting_classified if greeting_classified[s] == 'Bye'}
    hellos = {s.text: s.get('class') for s in nps_chat.xml_posts() if s.get('class') == 'Greet'}
    byes = {s.text: s.get('class') for s in nps_chat.xml_posts() if s.get('class') == 'Bye'}
    print('ready')


class Model:
    def __init__(self, hist, state_size=2):
        self.model = []
        self.state_size = state_size
        if type(hist) == str:
            hist = hist.split('\n')
        for s in hist:
            sent = word_tokenize(s)
            sent.insert(0, '__begin__')
            sent.insert(0, '__begin__')
            sent.append('__end__')
            s_model = []
            for i in range(state_size - 1, len(sent) - 1):
                state = sent[i - state_size + 1:i + 1]
                s_model.append([[a.lower() for a in state], sent[i + 1]])

            for p in s_model:
                new = True
                for m in self.model:
                    if m[0] == p[0]:
                        if p[1] in m[1]:
                            m[1][p[1]] += 1
                        else:
                            m[1][p[1]] = 1
                        new = False
                if new:
                    self.model.append([p[0], {p[1]: 1}])

    def make_sentence(self, seed='', threshold=0.4):
        if type(seed) == str:
            seed = word_tokenize(seed)
        sent = ['__begin__', '__begin__'] + seed
        while sent[-1] != '__end__':
            weights = {}
            for i in range(self.state_size, -1, -1):
                state = [a.lower() for a in sent[-i:]]
                # print('state: '+str(state))
                for s in self.model:
                    # print(s[0][-i:], state)
                    if [a.lower() for a in s[0][-i:]] == state:
                        for w in s[1]:
                            # print(w)
                            if w in weights:
                                weights[w] += s[1][w]
                            else:
                                weights[w] = s[1][w]
                if weights:
                    break
            # print(weights)
            counts = []
            words = []
            for w in weights:
                counts.append(weights[w])
                words.append(w)
            total = sum(counts)
            probs = [a/total for a in counts]
            draw = numpy.random.choice(words, 1, p=probs)[0]
            sent.append(draw)
            # print('sent: ' + str(sent))

        return detokenizer.detokenize(sent[2:-1])


def generate_relevant_sentence(vector_in, s):
    sentences = {}
    for i in range(0, 200):
        sentence = model.make_sentence(s)
        v_sentence = word_vectorize(sentence)
        sentences[sentence] = cosine_dic(vector_in, v_sentence)
    closest_out = nlargest(20, sentences, key=sentences.get)
    return choice(closest_out)


def get_response(m):
    vector_m = word_vectorize(m)
    t = parser.parse_one(word_tokenize(m))
    q_typ = classify_question(t)
    if q_typ == 'y/n-question':
        return choice(['yes', 'yup', 'uh-huh', 'no', 'nope', 'naw'])
    if q_typ == 'wh-question':
        wh_phrase = traverse_for(t, ['WHADJP', 'WHAVP', 'WHNP', 'WHPP', 'WHADVP'])
        wh_phrase = [w.lower() for w in wh_phrase]
        if 'who' in wh_phrase or 'whose' in wh_phrase or 'who\'s' in wh_phrase or 'whom' in wh_phrase:
            people = [w[0] for w in named_entities if w[1] == 'PERSON']
            return choice(people)
        if 'where' in wh_phrase:
            places = [w[0] for w in named_entities if w[1] == 'LOCATION']
            return choice(places)
        if 'when' in wh_phrase:
            times = [w[0] for w in named_entities if w[1] == 'DATE' or w[1] == 'TIME']
            return choice(times)
        if 'why' in wh_phrase:
            seeder = why_answer(m)
            return generate_relevant_sentence(vector_m, seeder)
        if 'how' in wh_phrase or 'what' in wh_phrase:
            seeder = uninvert(m)
            return generate_relevant_sentence(vector_m, seeder)
    g_typ = classify_greeting(m)
    print(g_typ)
    if g_typ == 'Greet':
        poss_hellos = {}
        for s in hellos:
            vector_s = word_vectorize(s)
            poss_hellos[s] = cosine_dic(vector_s, vector_m)
        largest = nlargest(10, poss_hellos, key=poss_hellos.get)
        return choice(largest)
    if g_typ == 'Bye':
        poss_byes = {}
        for s in hellos:
            vector_s = word_vectorize(s)
            poss_byes[s] = cosine_dic(vector_s, vector_m)
        largest = nlargest(10, poss_byes, key=poss_byes.get)
        return choice(largest)
    sims = {}
    for s in history:
        if not config['prefix'] in s:
            vector_s = word_vectorize(s)
            sims[s] = cosine_dic(vector_m, vector_s)
    # print(sims)
    largest = nlargest(10, sims, key=sims.get)
    # print(largest)
    seeders = sample(largest, k=5)
    sentences = {}
    for seeder in seeders:
        # print(seeder)
        seeder = word_tokenize(seeder)[:2]
        for i in range(0, 20):
            sentence = model.make_sentence(seeder)
            v_sentence = word_vectorize(sentence)
            sentences[sentence] = cosine_dic(vector_m, v_sentence)
    closest_out = nlargest(5, sentences, key=sentences.get)
    return choice(closest_out)


if __name__ == '__main__':
    print(find_question_root('What is the time??'))
    print(uninvert('What is the time?'))
