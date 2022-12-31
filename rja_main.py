from tqdm import tqdm
from mcmc import *
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
def rja(args):
    victim_classifier =victim_models(args.victim_model_name)
    mlm=MLM(10)
    dataset=MHDataset(args.dataset)
    dataloader=DataLoader(dataset,batch_size=1,shuffle=False)
    data_itr=tqdm.tqdm(enumerate(dataloader),
                       total=len(dataloader),
                       bar_format="{l_bar}{r_bar}"
    )
    results={
        'candidates':[],
        'success_overall':[],
        'best_adv':[],
    }
    torch.manual_seed(0)
    print('Finish class definining, and Start RJA')
    count=0
    save_path=args.save_dir+args.dataset+'_'+args.model_type
    for i,(text_id,label_id) in data_itr:
        # print(text_id[0])
        # print(label_id)
        # print(type(text_id))
        rja_run=RJA(
            text=text_id[0],
            label=label_id,
            NPW=args.NPW,
            classifier=victim_classifier,
            MLM=mlm,
            # hownet=Hownet,
            K=args.K,
            T=args.T,
            )
        print('Finish RJA definition, and Start RJA to run')
        adv_candidates, success_overall, best_adv=rja_run.run()
        count+=1
        results['candidates'].append(adv_candidates)
        results['success_overall'].append(success_overall)
        results['best_adv'].append(best_adv)
        print('########success_overall: '+str(success_overall)+'########')
        print('########best_adv:'+str(best_adv)+'########')
        # print(yaml.dump(best_adv, default_flow_style=False))
        print('########Success Attack Rate: '+str(count/(i+1))+'########')
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
    rja(args)