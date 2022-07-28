from torchtext.data import get_tokenizer
from torchtext.vocab import vocab
from collections import Counter, OrderedDict


def preprocessing_captions(data, max_qst_length):
    tokenizer = get_tokenizer("basic_english")
    vocabList = list()
    sentences = list()
    for line in data:
        for sentence in line[1]:
            vocabs = tokenizer(sentence)
            sentences.append(vocabs)
            for word in vocabs:
                vocabList.append(word)

    counter = Counter(vocabList)
    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    ordered_dict = OrderedDict(sorted_by_freq_tuples)
    dict = vocab(ordered_dict, min_freq=10, specials=["<unk>", "<start>", "<end>", "<pad>"])
    dict.set_default_index(dict["<unk>"])

    ans = list()
    for i in range(len(data)):
        line = data[i]
        captions_list = list()
        for sentence in line[1]:
            vocabs = tokenizer(sentence)
            number_of_words = list()
            for word in vocabs:
                number_of_words.append(dict[word])
            if len(number_of_words) > max_qst_length:
                continue
            while len(number_of_words) < max_qst_length:
                number_of_words.append(dict["<pad>"])
            captions_list.append(number_of_words)
        ans.append([line[0], captions_list])
    return ans, len(dict)
