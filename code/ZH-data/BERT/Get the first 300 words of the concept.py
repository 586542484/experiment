import json
import re

import pandas as pd
import codecs

url = '../pr_concepts.csv'

concepts = list(pd.read_csv(url, header=None)[0])

with codecs.open('概念content保存/pr_conceptContent.json', encoding='utf-8') as read_file:
    data = json.load(read_file)

IndexWords = []
for i in concepts:
   my_new_string = re.sub('[^\u4e00-\u9fa5]', '', data[i])

   needStringList = list(my_new_string)
   finalList1 = [(el.strip()) for el in needStringList]
   finalList = list(filter(None, finalList1))
   s = "".join(str(elem) for elem in finalList[:300])
   IndexWords.append(s)

df = pd.DataFrame({'IndexWords': IndexWords})
df.to_excel('概念前300词/pr_words-300.xlsx')
# print(IndexWords)
