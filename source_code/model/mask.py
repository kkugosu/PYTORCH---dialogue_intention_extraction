import torch
import numpy as np


def coder_mask(leng, maxsize, encoder):
    """

    make one-hot vector of mask from lengset
    :param leng:
    :param maxsize:
    :param encoder:
    :return:


        coder_mask(en_len)
        [[0. 0. 0. ... 0. 0. 1.]
         [0. 0. 0. ... 0. 0. 1.]
         [0. 0. 0. ... 0. 0. 1.]
         ...
         [0. 0. 0. ... 1. 0. 0.]
         [0. 0. 0. ... 1. 0. 0.]
         [0. 0. 0. ... 1. 0. 0.]]
        torch.sum(torch.tensor(encoder_mask(en_len)),1)
        1111111111...

    """

    var = np.zeros(shape=(len(leng), maxsize))  # len(leng) = BATCH_SIZE, leng[0]+1= largest dialogue + stop
    i = 0
    while i < len(leng):  # BATCH_SIZE
        j = 0
        while j < maxsize:
            if encoder & (j == leng[i] - 1):
                var[i][j] = 1
            if (not encoder) & (j < leng[i]):  # <= stop tag
                var[i][j] = 1

            j = j + 1
        i = i + 1
    var = torch.tensor(var)
    var = torch.unsqueeze(var, 2).type(torch.cuda.FloatTensor)
    return var
