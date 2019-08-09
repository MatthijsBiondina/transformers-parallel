from nltk.translate.bleu_score import sentence_bleu
import time
import torch
import torch.nn.functional as F
import warnings

from src.process.batch import create_masks
from src.process.translate import translate_preprocessed, translate_batch
from src.utils.tools import Tools as T


def train_epoch(model, opt, epoch, start, SRC, TRG):
    model.train()
    total_loss = 0

    for i, batch in T.poem(enumerate(opt.train),
                           description=f'epoch {epoch} (train)',
                           total=opt.train_len):
        src = batch.src.transpose(0, 1)
        trg = batch.trg.transpose(0, 1)
        with torch.no_grad():
            prd = translate_batch(src, trg, model, opt, SRC, TRG)

        # NO PEAK
        trg_input = trg[:, :-1]
        src_mask, trg_mask = create_masks(src, trg_input, opt)
        T.pyout(src.shape, trg_input.shape, src_mask.shape, trg_mask.shape)
        np_preds = model(src, trg_input, src_mask, trg_mask)

        # DO PEAK
        prd_input = prd[:, :-1]
        prd_mask = (prd != TRG.vocab.stoi['<pad>']).unsqueeze(-2)
        T.pyout(src.shape, prd_input.shape, src_mask.shape, prd_mask.shape)
        dp_preds = model(src, prd_input, src_mask, prd_mask)

        # CALCULATE LOSS
        ys = trg[:, 1:].contiguous().view(-1)
        opt.optimizer.zero_grad()
        np_loss = F.cross_entropy(
            np_preds.view(-1, np_preds.size(-1)), ys, ignore_index=opt.trg_pad)
        dp_loss = F.cross_entropy(
            dp_preds.view(-1, dp_preds.size(-1)), ys, ignore_index=opt.trg_pad)
        loss = np_loss + dp_loss

        # BACKPROP
        loss.backward()
        opt.optimizer.step()
        if opt.SGDR is True:
            opt.sched.step()

        total_loss += loss.item()

        # if (opt.checkpoint > 0 and
        #         ((time.time() - cptime) // 60) // opt.checkpoint >= 1):
        #     torch.save(model.state_dict(), './res/weights/model_weights')
        #     cptime = time.time()

    return total_loss / (i + 1)


def val_epoch(model, opt, epoch, start, c_epoch, c_loss):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, batch in T.poem(enumerate(opt.val),
                               description=f'epoch {epoch} (val)',
                               total=opt.val_len):
            src = batch.src.transpose(0, 1)
            trg = batch.trg.transpose(0, 1)

            trg_input = trg[:, :-1]
            src_mask, trg_mask = create_masks(src, trg_input, opt)
            preds = model(src, trg_input, src_mask, trg_mask)
            ys = trg[:, 1:].contiguous().view(-1)
            loss = F.cross_entropy(
                preds.view(-1, preds.size(-1)), ys, ignore_index=opt.trg_pad)
            total_loss += loss.item()
    avg_loss = total_loss / (i + 1)

    saved = False
    if avg_loss < c_loss:
        torch.save(model.state_dict(), './res/weights/model_weights')
        saved = True
        c_loss = avg_loss
        c_epoch = epoch

    return total_loss / (i + 1), c_epoch, c_loss, saved


def test_epoch(model, opt, SRC, TRG):
    model.eval()
    total_score = 0.

    for i, batch in T.poem(enumerate(opt.test),
                           description=f'calculating BLEU score',
                           total=len(opt.test)):
        src, trg = batch
        pred = translate_preprocessed(src, model, opt, SRC, TRG)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            total_score += sentence_bleu([trg], pred)

    return total_score / (i + 1)
