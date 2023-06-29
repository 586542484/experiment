import math

import pandas as pd
import numpy as np

from mediawiki import MediaWiki
wikipedia = MediaWiki()

# d = pd.read_csv(r'D:\Code\PycharmProjects\LK_project\LK_实验改进\词嵌入\AL-CPL\data_mining.csv', usecols=['A', 'B'], nrows=30)
d = pd.read_csv(r'D:\Code\PycharmProjects\LK_project\LK_实验改进\词嵌入\AL-CPL\physics.csv', usecols=['A', 'B'])
# concepts = d.values.tolist()[0:1000]
concepts = d.values.tolist()[1000:1962]
# print(concepts)
# print(len(concepts))
# print(wikipedia.page('DBSCAN').backlinks)
# print(len(wikipedia.page('DBSCAN').backlinks))

# 英文维基百科词条总数：6,641,740
PMI_count = []
count = 0
for i in concepts:
    A_backlinks = wikipedia.page(i[0]).backlinks
    B_backlinks = wikipedia.page(i[1]).backlinks
    result1 = len([x for x in A_backlinks if x not in B_backlinks])
    result2 = len([x for x in B_backlinks if x not in A_backlinks])
    result3 = len([x for x in A_backlinks if x in B_backlinks])
    p_c1 = (result1+1)/6641740
    p_c2 = (result2+1)/6641740
    p_c1c2 = (result3+1)/6641740
    pmi = ((np.log(p_c1) + np.log(p_c2))/np.log(p_c1c2))-1
    # fz = np.log(p_c1*p_c2)
    # fm = np.log(p_c1c2)
    # x = fz/fm
    if pmi <= 0:
        PMI = 0
    else:
        PMI = pmi
    PMI_count.append(PMI)
    count = count + 1
    print(count)

# 归一化处理
max_value = max(PMI_count)
min_value = min(PMI_count)
PMI_normal = [(i - min_value) / (max_value - min_value) for i in PMI_count]

df1 = pd.DataFrame(PMI_normal)
df1.to_excel("phy_PMI(part2).xlsx", header=None, index=None)

# print(PMI_count)
# print(PMI_normal)
