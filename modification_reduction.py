from fcntl import F_FULLFSYNC
from msilib.schema import Class
import random
import torch
import torch.nn.functional as F
import tqdm
import numpy as np
from utility import *
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import time
from metric import *
semantic=sentence_similarity()





class MH_MR:
    def __init__(self,text,ad_tokens,label,m,v,s,classifier,MLM,K,T):

        #Initialize the parameters
        self.input_text = text
        self.label = label
        self.classifier = classifier
        self.MLM = MLM
        self.K = K
        self.T = T
        self.tokenizer=word_tokenize
        self.detokenizer=TreebankWordDetokenizer()
        self.ad_tokens=ad_tokens
        self.tokens = self.tokenizer(self.input_text)
        self.reserve_prob=0.1
        self.ad_text=None
        self.candidates=[]
        # vairiables for RJA
        self.ori_v=v
        self.m=m
        self.v=v
        self.s=s
        #prepare the words candidates
        self.words_cans=[]
        for i in self.v:
            sys=get_synonyms(self.tokens[i])
            masked_sentence=copy.deepcopy(self.tokens)
            masked_sentence[i]='<mask>'
            masked_sentence=self.detokenizer.detokenize(masked_sentence)
            MLM_cans=self.MLM.cdts(masked_sentence,self.K)
            cans=sys+MLM_cans
            self.words_cans.append(cans)

        # candidate preparing
        saliencies=self.classifier.words_saliency(self.v,self.input_text,self.label)
        self.recover_dis=[]
        for ele in saliencies:
            dis=F.softmax(torch.tensor([0,ele]).float(),dim=0)
            self.recover_dis.append(dis)
    def target_dis(self,s,adv_pro):
        modification_rate=len(s)/len(self.tokens)
        pre_prob=self.classifier.prob(adv_pro,self.label)
        target_prob=(1-pre_prob)*(1-modification_rate)
        return target_prob.item()


    def propose_recover(self,):
        prop_v=[]
        prop_s=[]
        prob=1.0
        for i in range(len(self.recover_dis)):
            if random.random()>self.recover_dis[i][0]:
                prop_v.append(self.v[i])
                prop_s.append(self.s[i])
                prob=prob*self.recover_dis[i][1]
            else:
                prob=prob*self.recover_dis[i][0]
        adv_tokens=copy.deepcopy(self.tokens)
        for i in len(prop_s):
            adv_tokens[prop_v[i]]=prop_s[i]
        adv_text=self.detokenizer.detokenize(adv_tokens)
        return prop_v,prop_s,adv_tokens,adv_text,prob
    
    def rec_accept_prob(self,adv_text,prop_s,prop_prob):
        old_target=self.target_dis(self.s,self.input_text)
        new_target=self.target_dis(prop_s,adv_text)
        reverse_prob=copy.deepcopy(self.reserve_prob)
        accept_prob=new_target*reverse_prob/(old_target*prop_prob)
        u=random.random()
        acceptance=u<accept_prob
        return acceptance


    def propose_s(self):
        '''
        Propose the substitution of the victim word
        '''
        ##find the substitution from MLM
        v=copy.deepcopy(self.v)
        s=[]
        new_tokens=copy.deepcopy(self.tokens)
        for ele in v:
            i=self.ori_v.index(ele)
            words=copy.deepcopy(self.words_cans[i])
            sub=random.choice(words)
            new_tokens[ele]=sub
            s.append(sub)
            proposed_text=self.detokenizer.detokenize(new_tokens)
        return s,new_tokens,proposed_text
    def sub_update_acceptance(self,adv_text,prop_s):
        old_target=self.target_dis(self.s,self.input_text)
        new_target=self.target_dis(prop_s,adv_text)
        accept_prob=new_target/old_target
        u=random.random()
        acceptance=u<accept_prob
        return acceptance

   
    def mr(self):
        '''
        calculate the accept probability
        '''
        prop_v,prop_s,adv_tokens,adv_text,prob=self.propose_recover()
        acceptance=self.rec_accept_prob(adv_text,prop_s,prob)
        if acceptance:
            self.ad_tokens=adv_tokens
            self.adv=adv_text
            self.s=prop_s
            self.v=prop_v
            self.v_prob=prob
            self.acceptance=True
            self.ad_text=adv_text
        prop_s,ad_tokens,adv_text=self.propose_s()
        acceptance=self.sub_update_acceptance(adv_text,prop_s)
        if acceptance:
            self.ad_tokens=ad_tokens
            self.adv=adv_text
            self.s=prop_s
            self.acceptance=True
            self.ad_text=adv_text
            success=self.label!=self.classifier.predict(self.ad_text)

        d={
                'm':self.m,
                'v':self.v,
                's':self.s,
                'ad_text':self.ad_text,
                'ad_tokens':self.ad_tokens,
                'modification_rate':len(self.s)/len(self.tokens),
                'sem':semantic.sem(self.input_text,self.ad_text),
                'success':success,
                'acceptance':True
            }
        self.candidates.append(d)
    
    def run(self):
        start_time=time.time()
        suc_ind=[]
        suc_mr=[]
        for i in range(self.T):
            self.mr()
            if self.candidates[i]['success']==True:
                suc_ind.append(i)
                suc_mr.append(self.candidates[i]['modification_rate'])
        end_time=time.time()
        print('time of sampling the candidates:',end_time-start_time)
        success_overal=self.success>0
        if success_overal:
            best_ind=torch.max(torch.tensor(suc_mr),dim=0).indices
            best_ind=suc_ind[best_ind[0]]
            best_adv=self.candidates[best_ind]
        else:
            best_adv=False
        return self.candidates, success_overal, best_adv
    


