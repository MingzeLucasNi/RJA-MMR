from inspect import indentsize
import json

import yaml
s=[]
for i in range(10):
    s.append(i)
    with open('test.json','w') as f:
        json.dump(s,f)
    if i == 6:
        break
s
with open('test.json','r') as f:
        s=json.load(f)
s
d={
    'text':'I am from China',
    'tokens':['I', 'am', 'from', 'China'],
    'm':2,
    'v':[1,23,4],
    's':['i', 'am'],
    'ad_text':' I am a person',
    'ad_tokens':['I', 'am', 'a', 'person'],
    'sem':0.9,
    'success':10,
    'acceptance':True
}
import yaml
print(yaml.dump(d,default_flow_style=False,indent=5,width=4))
print(json.dumps(d,indent=5))
data=[1,2,3]
data[0:20]
import torch
import random
f=torch.arange(0,10).tolist()
f
v=[torch.tensor([1]).int(),torch.tensor([2]).int(),torch.tensor([3]).int()]
v
list(set(f)-set(v))
v=[random.randint(0,9) for i in range(2)]
v
f
list(set(f)-set(v))
q=[torch.arange(0,10),torch.arange(0,10)]
q=torch.to_json(q)
json.dumps(q)
torch.to_json()