import json
import re

import pandas as pd

url = r'D:\Code\PycharmProjects\LK_project\MHAVGAE-main\data\AL-CPL\dm_concepts.csv'

concepts = list(pd.read_csv(url, header=None)[0])

with open(r'D:\Code\PycharmProjects\LK_project\MHAVGAE-main\data\BERT\概念content保存\dm_conceptContent.json') as read_file:
    data = json.load(read_file)

IndexWords = []
for i in concepts:
   my_new_string = re.sub('[^a-zA-Z0-9 \n\.]', '', data[i])
   # print(my_new_string)
   indexList = my_new_string.split(' ')
   needStringList =[]
   for j in indexList:
      if len(j)!=0:
         needStringList.append(j.replace('.',''))
   finalList1 = [(el.strip()) for el in needStringList]
   finalList = list(filter(None, finalList1))
   # print(finalList[:500])
   s = " ".join(str(elem) for elem in finalList[:300])
   # print(s)
   IndexWords.append(s)

df = pd.DataFrame(
        {'IndexWords': IndexWords})
df.to_excel('概念前300词/dm_words-300.xlsx')
