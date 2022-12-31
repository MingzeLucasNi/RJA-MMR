'''
To make word-level attacks, we need to use the word-level tokenizer to extract the word inside the target text. The tokenizer will first encode the text into a sequence of sytax (words and symbols), after attacking it will decode the sequence back to a string. However, the decoding might be bothered with unless and semantically poor indentation and space, which could potentially affect the experiments. To solve this, we use the following codes to remove these meaningless indentations and space and make sure the decoding works perfectly on the experimental datasets.
'''
import torch
from nltk.tokenize import word_tokenize as tokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
detokenizer=TreebankWordDetokenizer()
def preprocess(data_name):
    data =torch.load('original_datasets/'+data_name+'.pt')
    for name in data.keys():
        for i in range(len(data[name]['feature'])):
            text=data[name]['feature'][i]
            seq=detokenizer.detokenize(tokenizer(text))
            data[name]['feature'][i]=seq
    return data
def reprocess(data):
    for name in data.keys():
        for i in range(len(data[name]['feature'])):
            text=data[name]['feature'][i]
            seq=detokenizer.detokenize(tokenizer(text))
            data[name]['feature'][i]=seq
    return data

def check_text(data):
    wrong_count =0
    index=[]
    for name in data.keys():
        for i in range(len(data[name]['feature'])):
            text=data[name]['feature'][i]
            seq=detokenizer.detokenize(tokenizer(text))
            if text!= seq:
                wrong_count+=1
                index.append([name,i])
    return wrong_count, index

data=preprocess('sst2')
data=reprocess(data)
count,index=check_text(data)

def clean(name):
    data=preprocess(name)
    for i in range(3):
        data=reprocess(data)
        wrong_count,_=check_text(data)
        if wrong_count==0:
            break
    torch.save(data,'datasets/'+name+'.pt')
    return data

data=clean('sst2')
check_text(data)
data=clean('emotion')
check_text(data)
data=clean('ag_news')
check_text(data)
##test if the data is one to one for the process of tokenizing and detokenizing
# for j in index:
#     text=data[j[0]]['feature'][j[1]]
#     seq=detokenizer.detokenize(tokenizer(text))
#     print(text)
#     print(seq)