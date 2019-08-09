from nltk.corpus import wordnet
import re
import sys
import torch
from torch.autograd import Variable


from src.opt.opt_translate import Opt
from src.models.transformer import get_model
from src.process.beam import beam_search
from src.process.process import Process as P
from src.utils.tools import Tools as T


def get_synonym(word, SRC):
    syns = wordnet.synsets(word)
    for s in syns:
        for l in s.lemmas():
            if SRC.vocab.stoi[l.name()] != 0:
                return SRC.vocab.stoi[l.name()]

    return 0


def multiple_replace(dict, text):
    regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

    return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text)


def debug(src, trg, out, SRC, TRG):
    src = src.cpu().numpy()
    trg = trg.cpu().numpy()
    out = out.cpu().numpy()
    s_phrase, t_phrase, o_phrase = [], [], []
    for s, t, o in zip(src, trg, out):
        s_phrase.append(' '.join([SRC.vocab.itos[tok] for tok in s]))
        t_phrase.append(' '.join([TRG.vocab.itos[tok] for tok in t]))
        o_phrase.append(' '.join([TRG.vocab.itos[tok] for tok in o]))
    ml_s = max(len(s) for s in s_phrase)
    ml_t = max(len(t) for t in t_phrase)
    ml_o = max(len(o) for o in o_phrase)
    for s, t, o in zip(s_phrase, t_phrase, o_phrase):
        T.pyout(s, ' ' * (ml_s - len(s)), ' | ',
                t, ' ' * (ml_t - len(t)), ' | ',
                o, ' ' * (ml_o - len(o)))


def translate_batch(src, trg, model, opt, SRC, TRG):
    model.eval()
    src_mask = (src != opt.src_pad).unsqueeze(-2)
    e_outputs = model.encoder(src, src_mask)
    out = torch.full(trg.shape, opt.trg_pad).long().to(opt.device)
    out[:, 0] = TRG.vocab.stoi['<sos>']
    for ii in range(1):  # trg.shape[1]):
        out_mask = (out != TRG.vocab.stoi['<pad>']).unsqueeze(-2)
        out_ = model.out(model.decoder(out, e_outputs, src_mask, out_mask))
        _, out_ = out_.max(-1)
        T.pyout(out_mask.shape)

    debug(src, trg, out, SRC, TRG)
    T.pyout(out_.shape, out.shape)
    model.train()


def translate_sentence(sentence, model, opt, SRC, TRG):
    model.eval()
    indexed = []
    T.pyout(sentence)
    sentence = SRC.preprocess(sentence)
    T.pyout(sentence)

    for tok in sentence:
        if SRC.vocab.stoi[tok] != 0 or opt.floyd is True:
            indexed.append(SRC.vocab.stoi[tok])
        else:
            indexed.append(get_synonym(tok, SRC))
    sentence = Variable(torch.LongTensor([indexed]))
    sentence = sentence.to(opt.device)

    sentence = beam_search(sentence, model, SRC, TRG, opt)
    T.pyout(sentence)
    sys.exit(0)

    return multiple_replace({' ?': '?', ' !': '!', ' .': '.', '\' ': '\'',
                             ' ,': ','}, sentence)


def translate_preprocessed(sentence, model, opt, SRC, TRG):
    indexed = []
    for tok in sentence:
        if SRC.vocab.stoi[tok] != 0 or opt.floyd is True:
            indexed.append(SRC.vocab.stoi[tok])
        else:
            indexed.append(get_synonym(tok, SRC))
    sentence = Variable(torch.LongTensor([indexed]))
    sentence = sentence.to(opt.device)

    sentence = beam_search(sentence, model, SRC, TRG, opt)
    return sentence.split(' ')


def translate(opt, model, SRC, TRG):
    sentences = opt.text.lower().split('.')
    translated = []

    for sentence in sentences:
        translated.append(translate_sentence(
            sentence + '.', model, opt, SRC, TRG).capitalize())

    return (' '.join(translated))


def main():
    opt = Opt('./res/weights', 'en', 'fr')
    opt.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    assert opt.k > 0
    assert opt.max_strlen > 10

    SRC, TRG = P.create_fields(opt)
    model = get_model(opt, len(SRC.vocab), len(TRG.vocab))

    while True:
        opt.text = input(
            "Enter a sentence to translate "
            "(type 'f' to load from file, or 'q' to quit):\n")
        if opt.text == "q":
            break
        if opt.text == "f":
            fpath = input(
                "Enter path to text file:\n")
            try:
                opt.text = ' '.join(
                    open(fpath, encoding='utf-8').read().split('\n'))
            except Exception:
                T.trace("error opening or reading text file", fpath)
        phrase = translate(opt, model, SRC, TRG)
        T.pyout('> ' + phrase + '\n')


if __name__ == '__main__':
    main()
