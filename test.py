english = []
french = []

with open('./res/data/europarl/train/english.txt', 'r') as en:
    with open('./res/data/europarl/train/french.txt', 'r') as fr:
        for en_line, fr_line in zip(en, fr):
            english.append(en_line.strip().split(' '))
            french.append(fr_line.strip().split(' '))

print(len(english), len(french), min(english, key=lambda x: len(x)),
      min(french, key=lambda x: len(x)))
