##import packages
import torch
from torch.utils.data import Dataset
import json
# from torch.utils.data import DataLoader
# from transformers import BertTokenizer, BertForMaskedLM



# tokenizer = BertTokenizer.from_pretrained( 'bert-base-uncased')
class MHDataset(Dataset):
    def __init__(self, data_name,cuda=False):
        data=torch.load('datasets/'+data_name+'.pt')['test']
        self.features = data['feature'][0:500]
        self.labels = data['label'][0:500]
        self.cuda = cuda
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        text_id = self.features[idx]
        label_id=self.labels[idx]
        return text_id, label_id
# dataset=MHDataset('ag_news')
# dataloader=DataLoader(dataset,batch_size=1,shuffle=False)
# for i, (text_id, label_id) in enumerate(dataloader):
#     print(text_id)
#     print(label_id)
#     break
class MHMR_Dataset(Dataset):
    def __init__(self, file_path,data_name,cuda=False):
        self.label=torch.load('datasets/'+data_name+'.pt')['test']['label'][0:500]
        self.feature=torch.load('datasets/'+data_name+'.pt')['test']['feature'][0:500]

        with open(file_path) as f:
            self.data = json.load(f)['best_adv']
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text_id = self.data[idx]
        feature= self.feature[idx]
        label_id=self.label[idx]
        return text_id, feature, label_id