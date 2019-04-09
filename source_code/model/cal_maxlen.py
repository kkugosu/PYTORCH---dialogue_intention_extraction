import numpy as np


def dialogue_maxlen_per_batch(len_info):
    return len_info[0]


def sentence_maxlen_per_batch(batch_data):
    i = 0
    maxlen = 1
    while i < len(batch_data):

        if sentence_maxlen_per_dialogue(batch_data[i]) > maxlen:
            maxlen = sentence_maxlen_per_dialogue(batch_data[i])
        i = i + 1
    return maxlen


def sentence_maxlen_per_dialogue(_data):
    """

    used in lamda function

    return longest sentence len per dialogue

    :param _data:
    :return:
    """
    i = 0
    maxleng = 0
    while i < len(_data.Text):  # len(data.Text) = dialogue length

        text, leng = sent_loader(_data.Text[i])
        if leng > maxleng:
            maxleng = leng
        i = i + 1
    return maxleng


def sent_loader(sentence):

    """
    pre_process per sentence

    :param sentence:
    :return:
    """

    result = []
    for elem in sentence.split(' '):
        if elem != '':
            result = np.append(result, elem)
    return result, len(result)
