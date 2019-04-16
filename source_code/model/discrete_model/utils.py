import numpy as np
import torch

def cal_accuracy(model_predict, real_tag):
    '''
    Args:
        model_predict
            model predicted tags
        real_tag
            real tags
        tag_len
            tag len

    Yields:
        accuracy

    Example:

    real = torch.tensor([  0,   2,   1,   2,   1,   1,   1,   1,   1,   2,   1,   1, 1,   1,  29], device='cuda:0')
    model = [2, 1, 2, 1, 17, 1, 17, 1, 3, 1, 17, 1, 17, 29]
    taglen = len(model)

    (npreal[tagseq+1]//4) real emotion
    (model_predict[tagseq]//4) model emotion

    (npreal[tagseq+1]%4) real action
    (model_predict[tagseq]%4) model action

    emotion err = 0.3076923076923077
    action err = 0.07692307692307693
    accuracy = 0.8076923076923077
    '''
    tag_len = len(model_predict)
    npreal = real_tag.cpu().numpy()
    tagseq = 0
    emotiontag = []
    actiontag = []
    emoerr = 0
    acterr = 0
    while tagseq < tag_len - 1:

        emotiontag = np.append(emotiontag, npreal[tagseq + 1] // 4)
        actiontag = np.append(actiontag, npreal[tagseq + 1] % 4)

        if (npreal[tagseq + 1] // 4) != (model_predict[tagseq] // 4):
            emoerr = emoerr + 1

        if (npreal[tagseq + 1] % 4) != (model_predict[tagseq] % 4):
            acterr = acterr + 1

        tagseq = tagseq + 1

    return 1 - (emoerr / tagseq + acterr / tagseq) / 2


def make_mask(leng):
    '''
    make one-hot vector of mask from lengset
    '''

    var = np.zeros(shape=(len(leng), leng[0]))  # len(leng) = BATCH_SIZE, leng[0]+1= largest dialogue + stop
    i = 0
    while i < len(leng):  # BATCH_SIZE
        j = 0
        while j < leng[0]:
            if j < leng[i]:  # <= stop tag
                var[i][j] = 1
            j = j + 1
        i = i + 1
    return var


def loss_filtering(loss_arr, filtering_value, newary, batchnum):
    """
    function for prevent overfitting

    Args:
        loss_arr:
            loss array for batch data

        filtering_value:
            allowed maximum loss

        newary:
            index for big loss data

        batchnum:
            current batch count


    Yields:
        loss_arr:
            filtered loss array for batch data

    """
    i = 0
    err_count = 0
    while i < len(loss_arr):
        if loss_arr[i] < filtering_value:
            loss_arr[i] = 0
        elif loss_arr[i] > (filtering_value * 4):
            err_count = err_count + 1
            newary.append(i + batchnum * 100)
        i = i + 1
    print("###############################################errcount", err_count)
    return loss_arr, newary


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