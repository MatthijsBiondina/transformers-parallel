import torch
import torch.nn.functional as F
import math

from src.process.batch import nopeak_mask
from src.utils.tools import Tools as T


def init_vars(src, model, SRC, TRG, opt):
    init_tok = TRG.vocab.stoi['<sos>']
    src_mask = (src != SRC.vocab.stoi['<pad>']).unsqueeze(-2)
    e_output = model.encoder(src, src_mask)

    outputs = torch.LongTensor([[init_tok]]).to(opt.device)
    trg_mask = nopeak_mask(1, opt)

    out = model.out(model.decoder(outputs, e_output, src_mask, trg_mask))

    out = F.softmax(out, dim=-1)
    probs, ix = out[:, -1].data.topk(opt.k)
    log_scores = torch.Tensor([math.log(prob)
                               for prob in probs.data[0]]).unsqueeze(0)

    outputs = torch.zeros(opt.k, opt.max_strlen).long().to(opt.device)
    outputs[:, 0] = init_tok
    outputs[:, 1] = ix[0]

    e_outputs = torch.zeros(opt.k, e_output.size(-2),
                            e_output.size(-1)).to(opt.device)
    e_outputs[:, :] = e_output[0]

    return outputs, e_outputs, log_scores


def k_best_outputs(outputs, out, log_scores, i, k):
    # get topk next words for each beam
    probs, ix = out[:, -1].data.topk(k)

    # for each of those next words, compute new score if added to
    # corresponding beam
    log_probs = torch.Tensor(
        [math.log(p) for p in probs.data.view(-1)]).view(k, -1) + \
        log_scores.transpose(0, 1)
    # from the k**2 newly created beams, select top k
    k_probs, k_ix = log_probs.view(-1).topk(k)

    row = k_ix // k  # which beam
    col = k_ix % k  # which next word in beam

    outputs[:, :i] = outputs[row, :i]  # words to t-1 from previous outputs
    outputs[:, i] = ix[row, col]  # words on t from current output

    log_scores = k_probs.unsqueeze(0)

    return outputs, log_scores


def beam_search(src, model, SRC, TRG, opt):
    try:
        outputs, e_outputs, log_scores = init_vars(src, model, SRC, TRG, opt)
        eos_tok = TRG.vocab.stoi['<eos>']
        src_mask = (src != SRC.vocab.stoi['<pad>']).unsqueeze(-2)
        ind = None
        for i in range(2, opt.max_strlen):
            trg_mask = nopeak_mask(i, opt)

            out = model.out(model.decoder(
                outputs[:, :i], e_outputs, src_mask, trg_mask))
            out = F.softmax(out, dim=-1)

            # T.pyout("> beam out", out.shape)

            outputs, log_scores = k_best_outputs(
                outputs, out, log_scores, i, opt.k)

            # Occurrences of end symbols for all input sentences.
            ones = (outputs == eos_tok).nonzero()
            sentence_lengths = torch.zeros(
                len(outputs), dtype=torch.long).to(opt.device)
            for vec in ones:
                i = vec[0]
                if sentence_lengths[i] == 0:  # First end symbol has not been found
                    # Position of first end symbol
                    sentence_lengths[i] = vec[1]

            num_finished_sentences = len(
                [s for s in sentence_lengths if s > 0])

            if num_finished_sentences == opt.k:
                alpha = 0.7
                div = 1 / (sentence_lengths.type_as(log_scores)**alpha)
                _, ind = torch.max(log_scores * div, 1)
                ind = ind.data[0]
                break

        if ind is None:
            try:
                length = (outputs[0] == eos_tok).nonzero()[0]  # <- error
            except IndexError:
                length = len(outputs[0])
            finally:
                return ' '.join(
                    [TRG.vocab.itos[tok] for tok in outputs[0][1:length]])
        else:
            try:
                length = (outputs[ind] == eos_tok).nonzero()[0]
            except IndexError:
                length = len(outputs[ind])
            finally:
                return ' '.join(
                    [TRG.vocab.itos[tok] for tok in outputs[ind][1:length]])
    except Exception as e:
        T.trace(src)
        T.trace(outputs)
        raise e
