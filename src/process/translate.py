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


def translate_batch(src, trg, model, opt):
    T.pyout(src.shape)


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
