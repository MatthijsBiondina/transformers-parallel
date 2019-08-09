import os


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
    k = 3

    def __init__(self,
                 root,
                 src_lang,
                 trg_lang):
        self.root = root
        self.src_train_data = os.path.join(root, 'train', src_lang[0] + '.txt')
        self.src_val_data = os.path.join(root, 'val', src_lang[0] + '.txt')
        self.src_test_data = os.path.join(root, 'test', src_lang[0] + '.txt')

        self.trg_train_data = os.path.join(root, 'train', trg_lang[0] + '.txt')
        self.trg_val_data = os.path.join(root, 'val', trg_lang[0] + '.txt')
        self.trg_test_data = os.path.join(root, 'test', trg_lang[0] + '.txt')

        self.src_lang = src_lang[1]
        self.trg_lang = trg_lang[1]
