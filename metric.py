from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn

class sentence_similarity:
    '''
    Usage: to calculate the sentence similarity
    '''
    def __init__(self, name='sentence-transformers/all-MiniLM-L6-v2'):
        self.Mini_model=SentenceTransformer(name) #import the Sentence embedding model
        self.Cos=nn.CosineSimilarity(dim=0, eps=1e-6) #define the cosine similarity function

    def sem(self,input,ref):
        '''
        input: the input sentence
        ref: the reference sentence
        '''
        example=[input,ref]
        embeddings=self.Mini_model.encode(example)
        similarity=self.Cos(torch.from_numpy(embeddings[0]), torch.from_numpy(embeddings[1]))
        return similarity