## Reversible Jump Attack to Textual Classifiers with Modification Reduction
Code for our paper "Reversible Jump Attack to Textual Classifiers with Modification Reduction".
### Requirements
This experiments are done based on the Huggingface (https://huggingface.co/) and pytorch. To set up the propoer environment, you may run the `requirements.txt` with the following command:
```
pip install requirements.txt 
```
or
```
conda env create -f RJA-MR.yml
```

### Usage
* There are three datasets: AG's News, Emotion, and SST2 in the [dataset] folder. If you wanna use your own data, please save them with torch .pt file.
* To perform the attack, you can run the following command:
```
python rja_main.py
python mr_main.py
```