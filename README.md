# HLAIImaster: a deep learning method with adaptive domain knowledge predicts HLA II neoepitope immunogenic responses

While significant strides have been made in predicting neoepitopes that trigger autologous CD4+ T cell responses, accurately identifying the antigen presentation by human leukocyte antigen (HLA) class II molecules remains a challenge. This identification is critical for developing vaccines and cancer immunotherapies. Current prediction methods are limited, primarily due to a lack of high-quality training epitope datasets and algorithmic constraints. To predict the exogenous HLA class II-restricted peptides across most of the human population, we utilized the mass spectrometry data to profile > 223,000 eluted ligands over HLA-DR, -DQ, and -DP alleles. Here, by integrating this data with peptide processing and gene expression, we introduce HLAIImaster, an attention-based deep learning framework with adaptive domain knowledge for predicting neoepitope immunogenicity. Leveraging diverse biological characteristics and our enhanced deep learning framework, HLAIImaster is significantly improved against existing tools in terms of positive predictive value across various neoantigen studies. Robust domain knowledge learning accurately identifies neoepitope immunogenicity, bridging the gap between neoantigen biology and the clinical setting and paving the way for future neoantigen-based therapies to provide greater clinical benefit. In summary, we present a comprehensive exploitation of the immunogenic neoepitope repertoire of cancers, facilitating the effective development of “just-in-time” personalized vaccines.

## Flowchart
<p align="center">
<img src="https://github.com/TomasYang001/HLAIImaster/blob/main/HLAIImaster.png" align="middle" height="80%" width="80%" />
</p>



## The environment of HLAIImaster
```
python==3.8.16
numpy==1.19.2
pandas==1.3.5
torch==1.12
scikit-learn>=0.24.2
pandas>=1.2.4
prettytable>=2.2.1
rdkit~=2021.03.2
yacs~=0.1.8
```

## Installation Guide
Clone this Github repo and set up a new conda environment. It normally takes about 10 minutes to install on a normal desktop computer.
```
# create a new conda environment
$ conda create --name HLAIImaster python=3.8.16
$ conda activate HLAIImaster

# install requried python dependencies
$ conda install pytorch==1.12 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
$ conda install -c conda-forge rdkit==2021.03.2
$ pip install -U scikit-learn
$ pip install yacs
$ pip install prettytable

# clone the source code of HLAIImaster
$ git clone https://github.com/
$ cd HLAIImaster
```

## Dataset description
In this paper, epitope presentation and immunogenicity data sources are used, which are freely downloaded from NetMHCIIpan-4.0 (https://services.healthtech.dtu.dk/services/NetMHCIIpan-4.0/), IEDB (https://www.iedb.org/) and MHCBN (https://webs.iiitd.edu.in/raghava/mhcbn/). 



## Run the sequence-only version as an example for epitope immunogenicity exploration
By default, you can run our model using the immunogenicity dataset with:
```sh
python main.py
```


# Acknowledgments
The authors sincerely hope to receive any suggestions from you!

Of note, each input data of HLAIImaster in our study is too large to upload. 
If readers want to further study the epitope immunogenicity prediction according to our study, please contact us at once. 
Email: YJ2197224605@163.com
