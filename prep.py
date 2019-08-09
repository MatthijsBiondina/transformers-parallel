import random
import os

from src.utils.tools import Tools as T

# SPLIT CUSTOM IN TRAIN, VAL AND TEST SET
IN_ROOT = './res/data/tiny'
OU_ROOT = './res/data/reduced'

lang1 = 'english.txt'
lang2 = 'french.txt'

train_folder = os.path.join(OU_ROOT, 'train')
val_folder = os.path.join(OU_ROOT, 'val')
test_folder = os.path.join(OU_ROOT, 'test')
for f in (train_folder, val_folder, test_folder):
    T.makedirs(f)

with open(os.path.join(IN_ROOT, lang1), 'r') as l1:
    with open(os.path.join(IN_ROOT, lang2), 'r') as l2:
        for i, _ in enumerate(zip(l1, l2)):
            pass
length = i
indices = list(range(i + 1))
random.shuffle(indices)

test_idxs = {ii: None for ii in indices[:len(indices) // 10]}
val_idxs = {ii: None for ii in indices[len(
    indices) // 10:2 * (len(indices) // 10)]}

with open(os.path.join(IN_ROOT, lang1), 'r') as l1:
    with open(os.path.join(IN_ROOT, lang2), 'r') as l2:
        with open(os.path.join(train_folder, lang1), 'w+') as l1_train:
            with open(os.path.join(train_folder, lang2), 'w+') as l2_train:
                with open(os.path.join(
                        val_folder, lang1), 'w+') as l1_val:
                    with open(os.path.join(
                            val_folder, lang2), 'w+') as l2_val:
                        with open(os.path.join(
                                test_folder, lang1), 'w+') as l1_test:
                            with open(os.path.join(
                                    test_folder, lang2), 'w+') as l2_test:
                                for ii, (l1_phrase, l2_phrase) in T.poem(
                                        enumerate(zip(l1, l2)), total=length):
                                    try:
                                        test_idxs[ii]
                                        l1_test.write(l1_phrase)
                                        l2_test.write(l2_phrase)
                                    except KeyError:
                                        try:
                                            val_idxs[ii]
                                            l1_val.write(l1_phrase)
                                            l2_val.write(l2_phrase)
                                        except KeyError:
                                            l1_train.write(l1_phrase)
                                            l2_train.write(l2_phrase)


# # SPLIT EUROPARL IN TRAIN AND VAL SET
# def europarl()
# OU_ROOT = './res/data/europarl'

# with open('./res/data/fr-en/europarl-v7.fr-en.en', 'r') as en:
#     with open('./res/data/fr-en/europarl-v7.fr-en.fr', 'r') as fr:
#         for i, _ in enumerate(zip(en, fr)):
#             pass

# length = i
# indices = list(range(i + 1))
# random.shuffle(indices)

# val_idxs = {ii: None for ii in indices[:len(indices) // 10]}

# with open('./res/data/fr-en/europarl-v7.fr-en.en', 'r') as en:
#     with open('./res/data/fr-en/europarl-v7.fr-en.fr', 'r') as fr:
#         with open(os.path.join(
#                 OU_ROOT, 'train', 'english.txt'), 'w+') as en_train:
#             with open(os.path.join(
#                     OU_ROOT, 'train', 'french.txt'), 'w+') as fr_train:
#                 with open(os.path.join(
#                         OU_ROOT, 'val', 'english.txt'), 'w+') as en_val:
#                     with open(os.path.join(
#                             OU_ROOT, 'val', 'french.txt'), 'w+') as fr_val:
#                         for i, (en_phrase, fr_phrase) in T.poem(
#                                 enumerate(zip(en, fr)), total=length):
#                             try:
#                                 val_idxs[i]
#                                 en_val.write(en_phrase)
#                                 fr_val.write(fr_phrase)
#                             except KeyError:
#                                 en_train.write(en_phrase)
#                                 fr_train.write(fr_phrase)
