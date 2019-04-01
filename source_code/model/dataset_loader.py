

def batchload(dataset, repeat, batchsize, data_seq):
    """

    load data as much as batch

    Args:

        dataset:
            data to load
        repeat:
            True if repeat load data
        batchsize:
            batchsize
        data_seq:
            order of data to load

    Yields:

        Batch data

    :param dataset:
    :param repeat:
    :param batchsize:
    :param data_seq:
    :return:

    """

    while True:
        i = batchsize
        while i <= len(data_seq):
            batch = []
            batch_seq = 0
            batchnum = data_seq[i - batchsize:i]

            while batch_seq < batchsize:
                batch.append(dataset[batchnum[batch_seq]])
                batch_seq = batch_seq + 1
            # print("batchnum = ",i)
            yield batch
            i = i + batchsize

        if not repeat:
            break