class Opt:
    no_cuda = False
    SGDR = False
    epochs = 2
    d_model = 512
    n_layers = 6
    heads = 8
    dropout = 0.1
    batch_size = 1500
    printevery = 100
    lr = 0.0001
    load_weights = None
    create_valset = False
    max_strlen = 80
    floyd = False
    checkpoint = 0

    def __init__(self,
                 src_data,
                 trg_data,
                 src_lang,
                 trg_lang):
        self.src_data = src_data
        self.trg_data = trg_data
        self.src_lang = src_lang
        self.trg_lang = trg_lang
