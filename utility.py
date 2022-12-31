import copy
import OpenHowNet
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import torch
from torch.nn import functional as F
from transformers import RobertaTokenizer, RobertaModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM
detokenizer=TreebankWordDetokenizer()
# # The example of using the tokenzier and detokenizer.
# detokenizer=TreebankWordDetokenizer()
# text="I am from China, but I' ve tired of cooking Chinese Cusine."
# words = tokenizer(text)
# sen=detokenizer.detokenize(words)

hownet_dict_advanced = OpenHowNet.HowNetDict(init_sim=True)
def get_synonyms(word):
    c=hownet_dict_advanced.get_nearest_words(word, language='en',K=10, merge=True)
    return c


# class hownet:
#     def __init__(self,K):
#         self.K=K
#         self.hownet_dict = OpenHowNet.HowNetDict()
#         self.hownet_dict_advanced = OpenHowNet.HowNetDict(init_sim=True)
#         self.en_words_list=self.hownet_dict.get_en_words()
        
#     def dict(self):
#         return self.en_words_list
#     def sys_possible(self,tokens):
#         '''
#         Input: list of string
#         return: list of indices of possible synsets
#         '''
#         indices_pos=[]
#         for i in range(len(tokens)):
#             if tokens[i] in self.en_words_list:
#                 indices_pos.append(i)
#         return indices_pos

#     def get_synonyms(self,word):
#         '''
#         Input: string of word, int
#         return: list of string
#         '''
#         sys=self.hownet_dict_advanced.get_nearest_words(word, language='en',K=self.K,merge=True)
#         return sys

class MLM:
    def __init__(self,K):
        self.K=K
        self.tokenizer=RobertaTokenizer.from_pretrained("roberta-large")
        self.model = AutoModelForMaskedLM.from_pretrained('roberta-large')
    def cdts(self,mask_text):
        '''
        input: mask_text(string), number of candidates K (int)
        return: list of candidates (list)
        '''
        mask_id=50264
        inputs=self.tokenizer.encode(mask_text, return_tensors="pt")
        mask_position=(inputs.squeeze()==mask_id).nonzero().squeeze()
        mask_logits=self.model(inputs).logits.squeeze()[mask_position]
        top_k=torch.sort(mask_logits,descending=True).indices[:self.K]
        cand=[]
        for id in top_k:
            c=self.tokenizer.decode([id])
            cand.append(c)
        return cand


class victim_models:
    def __init__(self,model_name):
        self.tokenizer=AutoTokenizer.from_pretrained(model_name)
        self.model=AutoModelForSequenceClassification.from_pretrained(model_name)
        self.detokenizer=TreebankWordDetokenizer()
        self.word_tokenize=word_tokenize
    
    def logits(self,text,label):
        inputs=self.tokenizer.encode(text, return_tensors="pt")
        outputs=self.model(inputs).logits.squeeze()
        logits=outputs[label]
        return logits

    def mr_words_saliency(self,v,text,label):
        tokens=self.word_tokenize(text)
        ori_logit=self.logits(text,label)
        saliency=[]
        for i in v:
            rem_text=copy.deepcopy(tokens)
            rem_text.pop(i)
            rem_logit=self.logits(self.detokenizer.detokenize(rem_text),label)
            s=ori_logit-rem_logit
            saliency.append(s)
        return saliency


    def words_saliency(self,text,label):
        tokens=self.word_tokenize(text)
        ori_logit=self.logits(text,label)
        saliency=[]
        for i in range(len(tokens)):
            rem_text=copy.deepcopy(tokens)
            rem_text.pop(i)
            rem_logit=self.logits(self.detokenizer.detokenize(rem_text),label)
            s=ori_logit-rem_logit
            saliency.append(s)
        return saliency

    def prob(self,text,label):
        inputs=self.tokenizer.encode(text, return_tensors="pt")
        outputs=self.model(inputs).logits.squeeze()
        prob=F.softmax(outputs,dim=0)[label]
        return prob

    def predict(self,text):
        inputs=self.tokenizer.encode(text, return_tensors="pt")
        outputs=self.model(inputs).logits.squeeze()
        prob=F.softmax(outputs,dim=0).squeeze()
        pre_label=torch.argmax(prob)
        return pre_label
