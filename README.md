# Reversible Jump Attack to Textual Classifiers with Modification Reduction (Machine Learning Journal 2024)
This repository contains Pytorch implementations of the Machine Learning Journal 2024 paper: Reversible Jump Attack to Textual Classifiers with Modification Reduction.
## Abstract
Recent advancements in NLP model vulnerabilities highlight the limitations of traditional adversarial example generation, often leading to detectable or ineffective attacks. Our study introduces the Reversible Jump Attack (RJA) and Metropolis-Hasting Modification Reduction (MMR) algorithms, aimed at crafting subtle yet potent adversarial examples. RJA broadens the search space for adversarial examples through randomization, optimizing the number of alterations. MMR then enhances these examples' stealthiness using the Metropolis-Hasting sampler. Our comprehensive experiments show RJA-MMR surpasses existing methods in attack efficacy, undetectability, and linguistic quality, marking a significant leap forward in adversarial research.

<p align="center">
<img src="https://github.com/MingzeLucasNi/RJA-MMR/blob/main/fig-flowchart.pdf" width=60% height=60%>
</p>

## Requirements

These experiments are done based on the Huggingface (https://huggingface.co/) and pytorch. To set up the proper environment, you may run the `requirements.txt` with the following command:
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

## Citation

When using this code, or the ideas of BU-MHS, please cite the following paper
<pre><code>@article{ni2024RJA,
  title={Reversible Jump Attack to Textual Classifiers with Modification Reduction},
  author={Mingze Ni, Zhensu Sun and Wei Liu},
  journal={Machine Learning},
  year={2024},
  publisher={Springer}
}

</code></pre>


## Contact

Please contact Mingze Ni at firstname.lastname@uts.edu.au or [Wei Liu](https://wei-research.github.io/) at firstname.lastname@uts.edu.au if you're interested in collaborating on this research!
