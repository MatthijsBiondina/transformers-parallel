import dill as pickle
import sys
import time
import torch

from src.loop.loop import train_epoch, val_epoch, test_epoch
from src.models.transformer import get_model
from src.process.optim import CosineWithRestarts
from src.process.process import Process as P
from src.opt.opt_main import Opt
from src.utils.tools import Tools as T


def train_model(model, opt, SRC, TRG):
    model.train()
    c_epoch, c_loss = 0, float('inf')
    start = time.time()
    opt.wfill = len(str(opt.epochs + 1))

    for epoch in T.poem(range(opt.epochs), description="training model..."):
        if epoch > c_epoch + 10:
            T.pyout(f"Early stopping after epoch {epoch}")
            break
        train_loss = train_epoch(model, opt, epoch, start, SRC, TRG)
        val_loss, c_epoch, c_loss, saved = val_epoch(
            model, opt, epoch, start, c_epoch, c_loss, SRC, TRG)
        T.pyout(
            "%dm: epoch %s%d  |  train loss = %.3f, eval loss = %.3f %s" %
            ((time.time() - start) // 60,
             ' ' * (opt.wfill - len(str(epoch + 1))),
             epoch + 1,
             train_loss,
             val_loss,
             "(model saved)" if saved else ""))
    bleu = test_epoch(model, opt, SRC, TRG)
    T.pyout("Training complete  -  BLEU score = %.3f" % (bleu,))


def test():
    pass


def main():
    opt = Opt('./res/data/reduced', ('english', 'en'), ('french', 'fr'))
    # opt.load_weights = './res/weights'
    opt.checkpoint = 60
    opt.printevery = 1
    opt.epochs = 2500
    opt.batch_size = 100

    opt.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")

    P.read_data(opt)
    SRC, TRG = P.create_fields(opt)
    opt.train, opt.val = P.create_dataset(opt, SRC, TRG)

    opt.test = P.create_testset(
        opt, SRC, TRG, opt.src_test_data, opt.trg_test_data)

    model = get_model(opt, len(SRC.vocab), len(TRG.vocab))

    opt.optimizer = torch.optim.Adam(
        model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)
    if opt.SGDR:
        opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)

    if opt.checkpoint > 0:
        T.pyout("model weights will be saved to directory ./res/weights/")

    if opt.load_weights is not None and opt.floyd is not None:
        T.makedirs('./res/weights')
        pickle.dump(SRC, open('./res/weights/SRC.pkl', 'wb'))
        pickle.dump(TRG, open('./res/weights/TRG.pkl', 'wb'))

    train_model(model, opt, SRC, TRG)

    pickle.dump(SRC, open('./res/weights/SRC.pkl', 'wb'))
    pickle.dump(TRG, open('./res/weights/TRG.pkl', 'wb'))

    torch.save(model.state_dict(), './res/weights/model_weights')


if __name__ == '__main__':
    main()
    T.pyout("SUCH WOW, MUCH SUCCES!")
