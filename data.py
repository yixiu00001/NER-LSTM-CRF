import sys, pickle, os, random
import numpy as np

## tags, BIO
#tag2label = {"O": 0,
#             "B-PER": 1, "I-PER": 2,
#             "B-LOC": 3, "I-LOC": 4,
#             "B-ORG": 5, "I-ORG": 6
#             }

tag2label = {"O": 0,
             "B-DISEASE": 1, "I-DISEASE": 2,
             "B-SYMPTOM": 3, "I-SYMPTOM": 4,
             "B-BODY": 5, "I-BODY": 6
             }

def read_corpus(corpus_path, vocab):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for line in lines:
        if line != '\n' and  len(line.strip().split())==2:
            [char, label] = line.strip().split()
            sent_.append(char)
            tag_.append(label)
        else:
            #sent_ = sentence2id(sent_, vocab)
            #label_ = [tag2label[tag] for tag in tag_]
            data.append((sent_, tag_))
            sent_, tag_ = [], []

    return data
def raw2Index(data, vocab):
    data_ = []
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, vocab)
        label_ = [tag2label[tag] for tag in tag_]

        data_.append((sent_, label_))
    return data_
        


def vocab_build(vocab_path, corpus_path, min_count):
    """
    BUG: I forget to transform all the English characters from full-width into half-width... 
    :param vocab_path:
    :param corpus_path:
    :param min_count:
    :return:
    """
    vocab_dir = vocab_path +  "word_voabulary.pkl"
    if os.path.exists(vocab_dir):
        with open(vocab_dir, 'rb') as data_f:
            vocabulary_word2index, vocabulary_index2word = pickle.load(data_f)
            return vocabulary_word2index, vocabulary_index2word
    else:
        if not os.path.exists(vocab_path):
            os.mkdir(vocab_path)
        data = read_corpus(corpus_path)
        
        vocabulary_word2index = {}
        vocabulary_index2word = {}
        for sent_, tag_ in data:
            for word in sent_:
                if word.isdigit():
                    word = '<NUM>'
                if word not in vocabulary_word2index:
                    vocabulary_word2index[word] = [len(vocabulary_word2index)+1, 1]
                else:
                    vocabulary_word2index[word][1] += 1
        low_freq_words = []
        for word, [word_id, word_freq] in vocabulary_word2index.items():
            if word_freq < min_count and word != '<NUM>':
                low_freq_words.append(word)
        for word in low_freq_words:
            del vocabulary_word2index[word]

        new_id = 1
        for word in vocabulary_word2index.keys():
            vocabulary_word2index[word] = new_id
            vocabulary_index2word[new_id] = word
            new_id += 1
        vocabulary_word2index['<UNK>'] = new_id
        vocabulary_word2index['<PAD>'] = 0
        vocabulary_index2word[new_id] = '<UNK>'
        vocabulary_index2word[0] = '<PAD>'

        with open(vocab_dir, 'wb') as fw:
            pickle.dump((vocabulary_word2index,vocabulary_index2word), fw)
            
        return vocabulary_word2index, vocabulary_index2word


def sentence2id(sent, word2id):
    """

    :param sent:
    :param word2id:
    :return:
    """
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id


def read_dictionary(vocab_path):
    """

    :param vocab_path:
    :return:
    """
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id


def random_embedding(vocab, embedding_dim):
    """

    :param vocab:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


def pad_sequences(sequences, pad_mark=0):
    """

    :param sequences:
    :param pad_mark:
    :return:
    """
    max_len = max(map(lambda x : len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


def batch_yield(data, batch_size, vocab, tag2label, shuffle=False):
    """

    :param data:
    :param batch_size:
    :param vocab:
    :param tag2label:
    :param shuffle:
    :return:
    """
    if shuffle:
        random.shuffle(data)
    
    seqs, labels = [], []
    #print("data=",len(data))
    for (sent_, tag_) in data:
        #sent_ = sentence2id(sent_, vocab)
        #label_ = [tag2label[tag] for tag in tag_]
        #print("tag=",tag_)
        if len(seqs) == batch_size:
            #print("now len=", len(seqs))
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(tag_)

    if len(seqs) != 0:
        yield seqs, labels

def batch_yieldOneIn(data, batch_size, vocab, tag2label, shuffle=False):
    """

    :param data:
    :param batch_size:
    :param vocab:
    :param tag2label:
    :param shuffle:
    :return:
    """
    if shuffle:
        random.shuffle(data)

    seqs, labels = [], []
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, vocab)
        label_ = [tag2label[tag] for tag in tag_]
        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(tag_)

    if len(seqs) != 0:
        yield seqs, labels
