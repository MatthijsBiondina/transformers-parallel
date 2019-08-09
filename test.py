import dill as pickle
import time
import torch
import torch.nn.functional as F

from src.models.transformer import get_model
from src.process.optim import CosineWithRestarts
from src.process.process import Process as P
from src.process.process import read_single_data_file
from src.process.batch import create_masks
from src.opt.opt_main import Opt
from src.utils.tools import Tools as T
from translate import multiple_replace


def train_model(model, opt):
    T.pyout("training model...")
    model.train()
    start = time.time()
    if opt.checkpoint > 0:
        cptime = time.time()

    for epoch in range(opt.epochs):
        total_loss = 0
        if opt.floyd is False:
            print("%dm: epoch %d [%s]  %d%%  loss = %s" %
                  ((time.time() - start) // 60,
                   epoch + 1,
                   "".join(' ' * 20),
                   0,
                   '...'), end='\r')

        if opt.checkpoint > 0:
            torch.save(model.state_dict(), './res/weights/model_weights')

        for i, batch in enumerate(opt.train):
            src = batch.src.transpose(0, 1)
            trg = batch.trg.transpose(0, 1)

            trg_input = trg[:, :-1]
            src_mask, trg_mask = create_masks(src, trg_input, opt)
            preds = model(src, trg_input, src_mask, trg_mask)
            ys = trg[:, 1:].contiguous().view(-1)
            opt.optimizer.zero_grad()
            loss = F.cross_entropy(
                preds.view(-1, preds.size(-1)), ys, ignore_index=opt.trg_pad)
            loss.backward()
            opt.optimizer.step()
            if opt.SGDR is True:
                opt.sched.step()

            total_loss += loss.item()

            if (i + 1) % opt.printevery == 0:
                p = int(100 * (i + 1) / opt.train_len)
                avg_loss = total_loss / opt.printevery
                if opt.floyd is False:
                    print("%dm: epoch %d [%s%s]  %d%%  loss = %.3f" %
                          ((time.time() - start) // 60,
                           epoch + 1, "".join('#' * (p // 5)),
                           "".join(' ' * (20 - (p // 5))),
                           p,
                           avg_loss), end='\r')
                else:
                    print("%dm: epoch %d [%s%s]  %d%%  loss = %.3f" %
                          ((time.time() - start) // 60,
                           epoch + 1, "".join('#' * (p // 5)),
                           "".join(' ' * (20 - (p // 5))),
                           p,
                           avg_loss))
                total_loss = 0
            if (opt.checkpoint > 0 and
                    ((time.time() - cptime) // 60) // opt.checkpoint >= 1):
                torch.save(model.state_dict(), './res/weights/model_weights')
                cptime = time.time()

        print("%dm: epoch %d [%s%s]  %d%%  loss = %.3f\nepoch %d complete,"
              " loss = %.03f" %
              ((time.time() - start) // 60,
               epoch + 1,
               "".join('#' * (100 // 5)),
               "".join(' ' * (20 - (100 // 5))),
               100,
               avg_loss,
               epoch + 1,
               avg_loss))


def test(model, opt, SRC, TRG):
    for i, batch in enumerate(opt.train):
        src = batch.src.transpose(0, 1)
        trg = batch.trg.transpose(0, 1)
        trg_input = trg[:, :-1]
        src_mask, trg_mask = create_masks(src, trg_input, opt)
        preds = model(src, trg_input, src_mask, trg_mask)
        T.pyout(src.shape, trg.shape,
                trg_mask.shape,
                preds.shape)
        T.pyout("")

        # src = src.cpu().numpy()
        # trg = trg.cpu().numpy()
        # for srcnums, trgnums in zip(src, trg):
        #     src_phrase = ' '.join([SRC.vocab.itos[tok] for tok in srcnums])
        #     trg_phrase = ' '.join([TRG.vocab.itos[tok] for tok in trgnums])

        #     src_phrase = multiple_replace({' ?': '?', ' !': '!', ' .': '.',
        #                                    '\' ': '\'', ' ,': ','}, src_phrase)
        #     trg_phrase = multiple_replace({' ?': '?', ' !': '!', ' .': '.',
        #                                    '\' ': '\'', ' ,': ','}, trg_phrase)

        #     T.pyout(src_phrase, ' | ', trg_phrase)


def main():
    opt = Opt('./res/data/reduced', ('english', 'en'), ('french', 'fr'))
    # opt.load_weights = './res/weights'
    opt.checkpoint = 60
    opt.printevery = 10
    opt.epochs = 25

    opt.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")

    P.read_data(opt)
    SRC, TRG = P.create_fields(opt)
    opt.train = P.create_dataset(opt, SRC, TRG)

    model = get_model(opt, len(SRC.vocab), len(TRG.vocab))

    opt.optimizer = torch.optim.Adam(
        model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)
    if opt.SGDR:
        opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)

    if opt.checkpoint > 0:
        T.pyout("model weights will be saved every %d minutes and at end of "
                "epoch to directory ./res/weights/" % (opt.checkpoint))

    if opt.load_weights is not None and opt.floyd is not None:
        T.makedirs('./res/weights')
        pickle.dump(SRC, open('./res/weights/SRC.pkl', 'wb'))
        pickle.dump(TRG, open('./res/weights/TRG.pkl', 'wb'))
    test(model, opt, SRC, TRG)
    # train_model(model, opt)

    # pickle.dump(SRC, open('./res/weights/SRC.pkl', 'wb'))
    # pickle.dump(TRG, open('./res/weights/TRG.pkl', 'wb'))

    # torch.save(model.state_dict(), './res/weights/model_weights')


if __name__ == '__main__':
    print(read_single_data_file('./res/data/europarl/train/english.txt'))
    # main()
    # T.pyout("SUCH WOW, MUCH SUCCES!")
