import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

d = pd.read_excel('../CH-ALCPL-非传递性-非对称性.xlsx', sheet_name='pr', usecols=['A', 'B'])
concepts = d.values.tolist()

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained("bert-base-chinese")

similarities = []
i = 0
for item in concepts:
    encoded_input1 = tokenizer(item[0], return_tensors='pt')
    output1 = model(**encoded_input1)
    list_bert1 = output1[0][:, 0, :].detach().numpy()[0].tolist()

    encoded_input2 = tokenizer(item[1], return_tensors='pt')
    output2 = model(**encoded_input2)
    list_bert2 = output2[0][:, 0, :].detach().numpy()[0].tolist()

    # 将 list_bert1 和 list_bert2 转换成二维数组
    arr1 = np.array(list_bert1).reshape(1, -1)
    arr2 = np.array(list_bert2).reshape(1, -1)
    similarity = cosine_similarity(arr1, arr2)
    normalized_similarity = (1 + similarity) / 2
    similarities.append(normalized_similarity)
    i = i+1
    print(i)
final_Similarities = [x.tolist()[0][0] for x in similarities]
# print(len(final_Similarities))
# print(final_Similarities)
df = pd.DataFrame(final_Similarities)
df.to_excel('pr_bert.xlsx', index=False, header=None)

