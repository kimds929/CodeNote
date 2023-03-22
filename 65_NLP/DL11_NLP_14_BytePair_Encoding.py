import _collections
import re

dictionary = {'l o w <end>':5, 'l o w e r <end>':2, 'n e w e s t <end>':6, 'w i d e s t <end>':3}

def get_pair(dictionary):
    pairs = _collections.defaultdict(int)
    for word, freq in dictionary.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq

    print('현재 pair:', dict(pairs))
    return pairs

def merge_dictionary(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair)) # e,s ->'e s' 공백을 확실히 넣어줌
    # 앞뒤가 공백인 것을 찾는다 -> 공백제거하고 텍스트찾기
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]

    return v_out

epochs = 10
for i in range(epochs):
    pairs = get_pair(dictionary)
    best = max(pairs, key=pairs.get)
    print('best pair:', best[0] + best[1])
    new_dictionary = merge_dictionary(best, dictionary)
    print('new_dictionary:', new_dictionary)

