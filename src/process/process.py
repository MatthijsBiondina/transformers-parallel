import dill as pickle
import os
import pandas as pd
import sys
from torchtext import data

from src.process.batch import MyIterator, batch_size_fn
from src.process.tokenize import Tokenize
from src.utils.tools import Tools as T


def read_single_data_file(path):
    try:
        return open(path,errors='replace').read().strip().split('\n')
    except FileNotFoundError:
        T.trace("error: '" + path + "' file not found", ex=1)


class Process:

    @staticmethod
    def read_data(opt):
        opt.src_train_data = read_single_data_file(opt.src_train_data)
        opt.src_val_data = read_single_data_file(opt.src_val_data)
        opt.src_test_data = read_single_data_file(opt.src_test_data)

        opt.trg_train_data = read_single_data_file(opt.trg_train_data)
        opt.trg_val_data = read_single_data_file(opt.trg_val_data)
        opt.trg_test_data = read_single_data_file(opt.trg_test_data)

    @staticmethod
    def create_fields(opt):
        spacy_langs = ['en', 'fr', 'de', 'es', 'pt', 'it', 'nl']
        if opt.src_lang not in spacy_langs:
            T.pyout('invalid src language:', opt.src_lang,
                    'supported languages:', spacy_langs)
        if opt.trg_lang not in spacy_langs:
            T.pyout('invalid trg language:', opt.trg_lang,
                    'supported languages:', spacy_langs)

        T.pyout("loading spacy tokenizers...")

        t_src = Tokenize(opt.src_lang)
        t_trg = Tokenize(opt.trg_lang)

        TRG = data.Field(lower=True, tokenize=t_trg.tokenizer,
                         init_token='<sos>', eos_token='<eos>')
        SRC = data.Field(lower=True, tokenize=t_src.tokenizer)

        if opt.load_weights is not None:
            try:
                T.pyout("loading presaved fields")
                SRC = pickle.load(open(f'{opt.load_weights}/SRC.pkl', 'rb'))
                TRG = pickle.load(open(f'{opt.load_weights}/TRG.pkl', 'rb'))
            except FileNotFoundError:
                T.trace(
                    "error opening SRC.pkl and TRG.pkl field files,",
                    "please ensure that they are in", opt.load_weights, "/",
                    ex=1)

        return SRC, TRG

    @staticmethod
    def create_testset(opt, SRC, TRG, src_data, trg_data):
        T.pyout("creating testset...")

        tok_data = [(SRC.preprocess(src_line), TRG.preprocess(trg_line))
                    for src_line, trg_line in zip(src_data, trg_data)]

        return tok_data

    @staticmethod
    def create_dataset(opt, SRC, TRG):
        T.pyout("creating dataset and iterator...")

        raw_data_t = {'src': [line for line in opt.src_train_data],
                      'trg': [line for line in opt.trg_train_data]}
        raw_data_v = {'src': [line for line in opt.src_val_data],
                      'trg': [line for line in opt.trg_val_data]}

        df_t = pd.DataFrame(raw_data_t, columns=["src", "trg"])
        df_v = pd.DataFrame(raw_data_v, columns=["src", "trg"])

        mask_t = (df_t['src'].str.count(' ') < opt.max_strlen) & (
            df_t['trg'].str.count(' ') < opt.max_strlen)
        mask_v = (df_v['src'].str.count(' ') < opt.max_strlen) & (
            df_v['trg'].str.count(' ') < opt.max_strlen)
        df_t = df_t.loc[mask_t]
        df_v = df_v.loc[mask_v]

        df_t.to_csv("translate_transformer_t.csv", index=False)
        df_v.to_csv("translate_transformer_v.csv", index=False)

        data_fields = [('src', SRC), ('trg', TRG)]
        train = data.TabularDataset('./translate_transformer_t.csv',
                                    format='csv', fields=data_fields)
        val = data.TabularDataset('./translate_transformer_v.csv',
                                  format='csv', fields=data_fields)

        train_iter = MyIterator(train,
                                batch_size=opt.batch_size,
                                device=opt.device,
                                repeat=False,
                                sort_key=lambda x: (len(x.src), len(x.trg)),
                                batch_size_fn=batch_size_fn,
                                train=True,
                                shuffle=True)
        val_iter = MyIterator(val,
                              batch_size=opt.batch_size,
                              device=opt.device,
                              repeat=False,
                              sort_key=lambda x: (len(x.src), len(x.trg)),
                              batch_size_fn=batch_size_fn,
                              train=False,
                              shuffle=False)

        os.remove('translate_transformer_t.csv')
        os.remove('translate_transformer_v.csv')

        if opt.load_weights is None:
            SRC.build_vocab(train)
            TRG.build_vocab(train)
            if opt.checkpoint > 0:
                T.makedirs('./res/weights')
                pickle.dump(SRC, open('./res/weights/SRC.pkl', 'wb'))
                pickle.dump(TRG, open('./res/weights/TRG.pkl', 'wb'))

        opt.src_pad = SRC.vocab.stoi['<pad>']
        opt.trg_pad = TRG.vocab.stoi['<pad>']

        opt.train_len = Process.get_len(train_iter)
        opt.val_len = Process.get_len(val_iter)

        return train_iter, val_iter

    @staticmethod
    def get_len(train):
        for i, b in enumerate(train):
            pass
        return i
