import pandas as pd
from transformers import BertTokenizer, BertModel

url = '概念前300词/pr_words-300.xlsx'
data = pd.read_excel(url)
col_1 = data["IndexWords"]
list1 = col_1.values.tolist()


# new_li = []
tinydict = {}

# a = 0
# for i in list1:
#     if i not in new_li:
#         new_li.append(i)
#         a = a+wLabel
#         tinydict[i] = a
# index = tinydict.keys()
# print(tinydict)

for index, item in enumerate(list1):
    tinydict[index] = item
print(tinydict.keys())

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained("bert-base-chinese")
for key, text in tinydict.items():
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    list_bert = output[0][:, 0, :].detach().numpy()[0].tolist()
    tinydict[key] = list_bert

# for text in tinydict:
#     encoded_input = tokenizer(text, return_tensors='pt')
#     output = model(**encoded_input)
#     list_bert = output[0][:, 0, :].detach().numpy()[0].tolist()
#     tinydict[text] = list_bert

df = pd.DataFrame(tinydict)
df2 = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)
df2.to_csv('feature_vector/pr-bert.csv', index=False)

# print(list1[0])
# list2 = list1[0].split(' ')
# print(list2)
# print(len(list2))
#
# con_list = []
#
# for i in range(0, len(list1)):
#     list2 = list1[i].split(' ')
#     si_list = []
#     for text in list2:
#         encoded_input = tokenizer(text, return_tensors='pt')
#         output = model(**encoded_input)
#         list_bert = output[0][:, 0, :].detach().numpy()[0].tolist()
#         si_list.append(list_bert)
#     con_list.append(si_list)
#
# print(len(con_list))
# print(con_list)


