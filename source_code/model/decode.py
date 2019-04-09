import torch


def decoding(decoder, decoder_hidden, maxlen, decoder_mask, batch_size):

    decoder_sent = 0

    seq = 0

    while decoder_sent < maxlen:  # all_output = seq*batch*hidden
        out, decoder_hidden = decoder(decoder_hidden, batch_size)
        decoder_sent = decoder_sent + 1

        if seq != 0:
            all_output = torch.cat((all_output, out), 1)

        else:
            all_output = out

        seq = seq + 1
    all_output = torch.mul(decoder_mask, all_output)

    return all_output