class Opt:
    k = 3
    max_strlen = 80
    d_model = 512
    n_layers = 6
    heads = 8
    dropout = 0.1
    no_cuda = False
    floyd = False

    def __init__(self, load_weights, src_lang, trg_lang):
        self.load_weights = load_weights
        self.src_lang = src_lang
        self.trg_lang = trg_lang
