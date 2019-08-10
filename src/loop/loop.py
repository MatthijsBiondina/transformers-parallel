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
        try:
            src = batch.src.transpose(0, 1)
            trg = batch.trg.transpose(0, 1)
            with torch.no_grad():
                prd = translate_batch(src, trg, model, opt, SRC, TRG)

            # NO PEAK
            trg_input = prd[:, :-1]
            src_mask, trg_mask = create_masks(src, trg_input, opt)
            np_preds = model(src, trg_input, src_mask, trg_mask)

            # DO PEAK
            prd_input = prd[:, :-1]
            prd_mask = (prd_input != TRG.vocab.stoi['<pad>']).unsqueeze(-2)
            prd_mask = torch.cat((prd_mask,) * prd_input.shape[-1], -2)
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
        except Exception:
            pass

        # if (opt.checkpoint > 0 and
        #         ((time.time() - cptime) // 60) // opt.checkpoint >= 1):
        #     torch.save(model.state_dict(), './res/weights/model_weights')
        #     cptime = time.time()

    return total_loss / (i + 1)


def val_epoch(model, opt, epoch, start, c_epoch, c_loss, SRC, TRG):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, batch in T.poem(enumerate(opt.val),
                               description=f'epoch {epoch} (val)',
                               total=opt.val_len):
            try:
                src = batch.src.transpose(0, 1)
                trg = batch.trg.transpose(0, 1)
                with torch.no_grad():
                    prd = translate_batch(src, trg, model, opt, SRC, TRG)

                # NO PEAK
                trg_input = trg[:, :-1]
                src_mask, trg_mask = create_masks(src, trg_input, opt)
                np_preds = model(src, trg_input, src_mask, trg_mask)

                # DO PEAK
                prd_input = prd[:, :-1]
                prd_mask = (prd_input != TRG.vocab.stoi['<pad>']).unsqueeze(-2)
                prd_mask = torch.cat((prd_mask,) * prd_input.shape[-1], -2)
                dp_preds = model(src, prd_input, src_mask, prd_mask)

                ys = trg[:, 1:].contiguous().view(-1)
                np_loss = F.cross_entropy(np_preds.view(
                    -1, np_preds.size(-1)), ys, ignore_index=opt.trg_pad)
                dp_loss = F.cross_entropy(dp_preds.view(
                    -1, dp_preds.size(-1)), ys, ignore_index=opt.trg_pad)
                total_loss += (np_loss.item() + dp_loss.item())
            except Exception:
                pass
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
