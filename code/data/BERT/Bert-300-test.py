import pandas as pd
from transformers import BertTokenizer, BertModel

url = '概念前300词/dm_words-300.xlsx'
data = pd.read_excel(url)
col_1 = data["IndexWords"]
list1 = col_1.values.tolist()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")
tinydict = {}

for index, text in enumerate(list1):
    encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    output = model(**encoded_input)
    list_bert = output.last_hidden_state[:, 0, :].detach().numpy()[0].tolist()
    tinydict[index] = list_bert

df = pd.DataFrame(tinydict)
df2 = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)
df2.to_csv('feature_vector/dm-bert-test.csv', index=False)
