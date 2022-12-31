from copy import deepcopy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import numpy as np
from utility import *
from metric import *
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import time
from multiprocessing import Pool
import os
semantic=sentence_similarity()



# random.randint(0,3)

# s=[1,3,3]
# s.pop(0)
# s
# if 3 in s:
#     s.remove(3)
# s
# torch.tensor([[0,0]]).shape
# F.softmax(torch.tensor([[0,0,1]]).float(),dim=0)
# if 2>=2:
#     print('yes')
# import numpy as np
# numbers = np.array([1,2,3,4,5,6,7,8,9])
# A = np.array([])
# a=[]
# b=[1,2,3]
# c=[]
# a.append(b)
# a.append(c)
# a
# list(set(numbers)- set(A))

# torch.range(0, 4).int()
# torch.range(1, 4, 0.5)
# x={
#     # 'text':[],
#     'NP':[],
#     'sub':[],
#     'Sem':[],
#     'MR':[],
#     # 'true_label':[],
#     'pred_label':[],
#     # 'true_prob_unattacked':[],
#     'true_prob_attacked':[]
#     }


class RJA:
    def __init__(self,text,label,NPW,classifier,MLM,K,T):

        #Initialize the parameters
        self.num_cpu=os.cpu_count()-1
        self.input_text = text
        self.label = label
        self.NPW = NPW
        self.classifier = classifier
        self.MLM = MLM
        # self.hownet = hownet
        self.K = K
        self.T = T
        self.tokenizer=word_tokenize
        self.detokenizer=TreebankWordDetokenizer()
        self.tokens=self.tokenizer(self.input_text)
        self.NPW=min(self.NPW,len(self.tokens)) #the number of words to be substituted
        # vairiables for RJA
        self.m=0
        self.v=[]
        self.s=[]
        self.can=text
        self.can_tokens=copy.deepcopy(self.tokens)
        self.candidates=[]
        self.success=0

        # candidate preparing
        self.saliencies=self.classifier.words_saliency(self.input_text,self.label)
        self.syn_can=[]
        self.mlm_can=[]
        self.sys_attackable=[]
        start_time=time.time()
        for i in range(len(self.tokens)):
            masked_sentence=copy.deepcopy(self.tokens)
            masked_sentence[i]='<mask>'
            masked_sentence=self.detokenizer.detokenize(masked_sentence)
            MLM_cans=self.MLM.cdts(masked_sentence)
            if self.tokens[i] not in MLM_cans:
                MLM_cans=MLM_cans[:-1]
            elif self.tokens[i] in MLM_cans:
                MLM_cans.remove(self.tokens[i])
            self.mlm_can.append(MLM_cans)
        with Pool(self.num_cpu) as pool:
            self.syn_can=pool.map(get_synonyms,self.tokens)
        for i in range(len(self.tokens)):
            self.sys_attackable.append(self.syn_can[i]!=[])

        
        end_time=time.time()
        print('time of preparing candidates',end_time-start_time)
    def propose_m(self):
        '''
        Note!!! the m should not be used with self.m directely because the m is changed in the function!!!!!!!!!!!!!!!!!!
        propose the m
        '''
        m=copy.deepcopy(self.m)
        v=copy.deepcopy(self.v)
        if m==0:
            m=1
            prob=1
        elif m==self.NPW:
            m=m-1
            prob=1
        else:
            s=sum(self.saliencies)
            s_0=0
            for ele in v:
                s_0+=self.saliencies[ele]
            s_1=s-s_0
            logits=[s_0,s_1]
            dis=F.softmax(torch.tensor(logits).float(),dim=0)
            a=torch.multinomial(dis,1)
            if a==0:
                m-=1
                prob=dis[0]
            else:
                m+=1
                prob=dis[1]

        return m,prob



    def propose_v(self,m):
        '''
        Input: m
        Output: v
        Description: propose the v
        '''
        v=copy.deepcopy(self.v)
        o_v=copy.deepcopy(self.v)
        # s=copy.deepcopy(self.s)


        if m==self.m-1:
            sel_v_ind=random.randint(0,self.m-1)
            v.pop(sel_v_ind)
            pop_ind=sel_v_ind
            sel_v_ind=o_v[sel_v_ind]
            prob=1/len(o_v)

        elif m==self.m+1:
            words_in=torch.arange(0,len(self.tokens)).tolist()
            unattack_ind=list(set(words_in)-set(v))
            sel_v_ind=random.randint(0,len(unattack_ind)-1)
            sel_v_ind=unattack_ind[sel_v_ind]
            v.append(sel_v_ind)
            prob=1/len(unattack_ind)
            pop_ind=False
        return v,sel_v_ind,pop_ind,prob

    def propose_s(self,m, v,v_ind,pop_ind):
        '''
        Propose the substitution of the victim word
        '''
        ##find the substitution from MLM
        s=copy.deepcopy(self.s)
        masked_tokens=copy.deepcopy(self.tokens)
        if m==self.m+1:
            MLM_cans=copy.deepcopy(self.mlm_can[v_ind])
            Syn_cans=copy.deepcopy(self.syn_can[v_ind])
            if self.sys_attackable[v_ind]==True:
                union=list(set(MLM_cans).union(set(Syn_cans)))
                intersect=list(set(MLM_cans).intersection(set(Syn_cans)))
                rest=list(set(union)-set(intersect))
                k=len(intersect)
                if random.random()<=k/self.K:
                    sub=random.choice(intersect)
                    s.append(sub)
                    prob=1/self.K
                else:
                    sub=random.choice(rest)
                    s.append(sub)
                    prob=1/(2*self.K)
            else:
                sub=random.choice(MLM_cans)
                s.append(sub)
                prob=1/self.K
        else:
            sub=masked_tokens[v_ind]
            s.pop(pop_ind)
            prob=1
        for ind in range(len(v)):
            masked_tokens[v[ind]]=s[ind]
        # masked_tokens[v_ind]=sub
        proposed_tokens=masked_tokens
        proposed_adv=self.detokenizer.detokenize(masked_tokens)
        
        return s,proposed_tokens,proposed_adv,prob
    
    def reverse_prob_m(self, m, v):
        if m==self.m-1:
            if m==0:
                reverse_prob=1
            else:
                s=sum(self.saliencies)
                s_0=0
                for ele in v:
                    s_0+=self.saliencies[ele]
                s_1=s-s_0
                logits=[s_0,s_1]
                dis=F.softmax(torch.tensor(logits).float(),dim=0)
                reverse_prob=dis[1]
        elif m==self.m+1:
            if m==self.NPW:
                reverse_prob=1
            else:
                s=sum(self.saliencies)
                s_0=0
                for ele in v:
                    s_0+=self.saliencies[ele]
                s_1=s-s_0
                logits=[s_0,s_1]
                dis=F.softmax(torch.tensor(logits).float(),dim= 0)
                reverse_prob=dis[0]
        return reverse_prob

    def reverse_prob_v(self,m):
        if m==self.m+1:
            reverse_prob=1/m
        elif m==self.m-1:
            reverse_prob=1/(len(self.tokens)-m)
        return reverse_prob
    # def reverse_prob_s(self):
    #     '''
    #     reverse the probability of the substitution is a constant 1/K because the both of the MLM_cans
    #     and Syn_cans sets include the original word, which mathematicaly means the reverse probability
    #     is a constant 1/K.
    #     '''
    #     return 1/self.K
    
    def target_prob(self,adv_pro):
        sem=semantic.sem(self.input_text,adv_pro)
        pre_prob=self.classifier.prob(adv_pro,self.label)
        target_prob=(1-pre_prob)*sem
        return target_prob,sem
    def rja(self):
        '''
        calculate the accept probability
        '''
        # proposing probability
        m=copy.deepcopy(self.m)
        propose_m,m_prob=self.propose_m()
        v,v_ind,pop_ind,v_prob=self.propose_v(propose_m)
        s,proposed_tokens,proposed_adv,s_prob=self.propose_s(propose_m,v,v_ind,pop_ind)

        proposing_prob=m_prob*v_prob

        # reverse proposing probability
        reverse_prob_m=self.reverse_prob_m(propose_m,v)
        reverse_prob_v=self.reverse_prob_v(propose_m)
        reverse_prob=reverse_prob_m*reverse_prob_v

        # target probability
        target_proposed,new_sem=self.target_prob(proposed_adv)
        target_ori,old_sem=self.target_prob(self.can)

        # acceptance probability
        if self.m==0:
            accept_prob=1
        elif self.m==self.NPW:
            accept_prob=1
        else:
            accept_prob=target_proposed*reverse_prob/target_ori/proposing_prob
        u=random.random()
        if u<accept_prob:
            self.m=propose_m
            self.v=v
            self.s=s
            self.can=proposed_adv
            self.can_tokens=proposed_tokens
            success=self.label!=self.classifier.predict(self.can)
            self.success=success+self.success
            d={
                'text':self.input_text,
                'tokens':self.tokens,
                'm':copy.deepcopy(self.m),
                'v':copy.deepcopy(self.v),
                's':copy.deepcopy(self.s),
                'ad_text':self.can,
                'ad_tokens':self.can_tokens,
                'sem':new_sem.item(),
                'success':success.item(),
                'acceptance':True
            }
        else:
            success=self.label!=self.classifier.predict(self.can)
            self.success=success+self.success
            d={
                'text':self.input_text,
                'tokens':self.tokens,
                'm':copy.deepcopy(self.m),
                'v':copy.deepcopy(self.v),
                's':copy.deepcopy(self.s),
                'ad_text':self.can,
                'ad_tokens':self.can_tokens,
                'sem':old_sem.item(),
                'success':success.item(),
                'acceptance':False
            }
            
        self.candidates.append(d)
        return d
    
    def run(self):
        start_time=time.time()
        suc_ind=[]
        suc_sem=[]
        for i in range(self.T):
            rel=self.rja()
            if rel['success']==1:
                suc_ind.append(i)
                suc_sem.append(self.candidates[i]['sem'])
        end_time=time.time()
        print('time of sampling the candidates:',end_time-start_time)
        success_overal=self.success>0
        if success_overal:
            best_ind=torch.max(torch.tensor(suc_sem),dim=0).indices
            best_ind=suc_ind[best_ind]
            best_adv=self.candidates[best_ind]
        else:
            best_adv=False
        return self.candidates, success_overal.item(), best_adv
