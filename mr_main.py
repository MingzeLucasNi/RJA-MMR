from plistlib import load
from tqdm import tqdm
from modification_reduction import *
from utility import *
from metric import *
from dataload import *
from torch.utils.data import DataLoader
import torch
import argparse
import json
import os
import yaml
os.environ["TOKENIZERS_PARALLELISM"] = "true"
print('finish loading the packages...')
def mr_iter(args):
    victim_classifier =victim_models(args.victim_model_name)
    mlm=MLM(10)
    load_path=args.save_dir+args.dataset+'_'+args.model_type+'.json'
    dataset=MHMR_Dataset(load_path)
    dataloader=DataLoader(dataset,batch_size=1,shuffle=False)
    data_itr=tqdm.tqdm(enumerate(dataloader),
                       total=len(dataloader),
                       bar_format="{l_bar}{r_bar}"
    )
    results={
        'candidates':[],
        'success_overall':[],
        'final_adv':[],
    }
    torch.manual_seed(0)
    print('Finish class definining, and Start MHMR')
    count=0
    for i,(best_rja, text, label) in data_itr:
        # print(text_id[0])
        # print(label_id)
        # print(type(text_id))
        if best_rja['best_adv']==False:
            results['candidates'].append(False)
            results['success_overall'].append(False)
            results['best_adv'].append(False)
        else:
            mhmr_run=MH_MR(
                text,
                best_rja['ad_tokens'],
                label,
                best_rja['m'],
                best_rja['v'],
                best_rja['s'],
                victim_classifier,
                mlm,
                args.K,
                args.T
                )
            adv_candidates, success_overall, best_adv=mhmr_run.run()
            results['candidates'].append(adv_candidates)
            results['success_overall'].append(success_overall)
            results['best_adv'].append(best_adv)
        print('########success_overall: '+str(success_overall)+'########')
        print('########best_adv:'+str(best_adv)+'########')
        # print(yaml.dump(best_adv, default_flow_style=False))
        # print('########Success Attack Rate: '+str(count/(i+1))+'########')
        save_path=args.save_dir+args.dataset+'_'+args.model_type+'_final'
        with open(save_path+'.json', 'w') as f:
            json.dump(results, f)
        # torch.save(results, save_path+'.pt')
    torch.save(results, save_path+'.pt')

def args_parser():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='emotion')
    parser.add_argument('--model_type', type=str, default='BERT_Classifier')
    parser.add_argument('--T', type=int, default=2000)#the number of iterations
    parser.add_argument('--K', type=int, default=10)#the number of substitution candidates
    parser.add_argument('--NPW', type=int, default=5)#the number of words in the NPW
    if parser.parse_args().dataset=='ag_news':
        parser.add_argument('--victim_model_name', type=str, default='textattack/bert-base-uncased-ag-news')
    elif parser.parse_args().dataset=='emotion':
        parser.add_argument('--victim_model_name', type=str, default='bhadresh-savani/distilbert-base-uncased-emotion')
    else:
        parser.add_argument('--victim_model_name', type=str, default='textattack/roberta-base-SST-2')
    '''
    victim_model_name:
        a. AG NEWS: textattack/bert-base-uncased-ag-news
        b. emotion: bhadresh-savani/distilbert-base-uncased-emotion
        b. SST-2: textattack/roberta-base-SST-2
    '''

    parser.add_argument('--save_dir', type=str, default='results/')

    return parser.parse_args()

if __name__ == '__main__':
    args=args_parser()
    mr_iter(args)