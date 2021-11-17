
import pandas as pd
from copy import deepcopy
from collections import Counter


def vote_result(votes):
    c = dict(Counter(votes))
    
    maxx = max(c.values())

    choices = []
    for key, value in c.items():
        if value == maxx:
            choices.append(key)

    if votes[-2] in choices:
        ret = votes[-2]
    else:
        ret = choices[0]

    if len(c) > 1:
        print(votes, ret, choices, end='\n' if ret == votes[-2] else '*\n')
    return ret



model_list = ['ensemble{}'.format(i) for i in range(10, 16)]
data_list = []

for model in model_list:
    data_list.append(pd.read_csv('results/{}_output.csv'.format(model)))

bak = deepcopy(data_list[0])

for i in range(len(bak['ID'])):
    votes = [data['emotion'][i] for data in data_list]
    res = vote_result(votes)
    bak['emotion'][i] = res

bak.to_csv('results/elected1.csv', index=False)



