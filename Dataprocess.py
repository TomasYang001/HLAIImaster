import os
import re
import pandas as pd
from io import StringIO
import numpy as np

paired = 'F'
hla_dict = {}
allele_dict = {}

pseu = pd.read_csv('/Users/tomasyang/Downloads/NetMHCIIpan_train/pseudosequence.2016.all.X.dat', sep='\t', header=None)
allele = pd.read_csv('/Users/tomasyang/Desktop/AlleleMap.csv', header=None)

alle1 = allele[0]
alle2 = allele[1]

for i in range(len(alle1)):
    allele_dict[str(alle1[i])] = str(alle2[i])

alle = pseu[0]
pseuseq = pseu[1]

for i in range(len(pseu)):
    hla_dict[str(alle[i])] = str(pseuseq[i])

Data = pd.read_csv('./HLAII/test1_com.csv')

Data = Data[:200000]

Data = Data.sample(frac=1.0).reset_index(drop=True)

Train = Data[:150000]
Test = Data[150000:]

def preprocess(dataset):
    #Preprocess TCR files
    # print('Processing: '+filedir)
    # if not os.path.exists(filedir):
    #     print('Invalid file path: ' + filedir)
    #     return 0
    # dataset = pd.read_csv(filedir, header=0)
    #Preprocess HLA_antigen files
    #remove HLA which is not in HLA_seq_lib; if HLA*01:01 not in HLA_seq_lib; then the first HLA startswith input HLA allele will be given
    #Remove antigen that is longer than 15aa
    if paired=='F':
        HLA_antigen=dataset[['HLA','Antigen']].dropna()
        HLA_list=list(HLA_antigen['HLA'])
        antigen_list=list(HLA_antigen['Antigen'])
        ind=0
        index_list=[]
        for i in HLA_list:
            if len([hla_allele for hla_allele in hla_dict.keys() if hla_allele.startswith(str(i))])==0:
                index_list.append(ind)
            ind=ind+1
        HLA_antigen=HLA_antigen.drop(HLA_antigen.iloc[index_list].index)
        HLA_antigen=HLA_antigen[HLA_antigen.Antigen.str.len()<26]
#         print(str(max(HLA_antigen.index)-HLA_antigen.shape[0])+' antigens longer than 15aa are dropped!')
        antigen_list=list(HLA_antigen['Antigen'])
        HLA_list=list(HLA_antigen['HLA'])
        label_list = dataset['label'].tolist()
    else:
        dataset=dataset.dropna()
        HLA_list=list(dataset['HLA'])
        ind=0
        index_list=[]
        for i in HLA_list:
            if len([hla_allele for hla_allele in hla_dict.keys() if hla_allele.startswith(str(i))])==0:
                index_list.append(ind)
                print('drop '+i)
            ind=ind+1
        dataset=dataset.drop(dataset.iloc[index_list].index)
        dataset=dataset[dataset.Antigen.str.len()<26]
        print(dataset.index)
#         print(str(max(dataset.index)-dataset.shape[0])+' antigens longer than 25aa are dropped!')
        antigen_list=dataset['Antigen'].tolist()
        HLA_list=dataset['HLA'].tolist()
        label_list = dataset['label'].tolist()
    return antigen_list,HLA_list, label_list


aa_dict_atchley=dict()
# with open(aa_dict_dir,'r') as aa:
#     aa_reader=csv.reader(aa)
#     next(aa_reader, None)
#     for rows in aa_reader:
#         aa_name=rows[0]
#         aa_factor=rows[1:len(rows)]
#         aa_dict_atchley[aa_name]=np.asarray(aa_factor,dtype='float')
########################### One Hot ##########################
aa_dict_one_hot = {'A': 0,'C': 1,'D': 2,'E': 3,'F': 4,'G': 5,'H': 6,'I': 7,'K': 8,'L': 9,
           'M': 10,'N': 11,'P': 12,'Q': 13,'R': 14,'S': 15,'T': 16,'V': 17,
           'W': 18,'Y': 19,'X': 20}  # 'X' is a padding variable
########################### Blosum ##########################
Caphla_MATRIX = pd.read_table(StringIO(u"""                                                                                      
	L	F	I	M	V	W	C	Y	H	A	T	G	P	R	Q	S	N	E	D	K	X	*
L	-0.274	-0.092	-0.181	-0.116	-0.214	-0.051	-0.164	-0.142	-0.012	-0.263	0.028	0.092	0.245	-0.092	-0.077	0.02	0.139	0.016	0.169	0.043	0.5	-1
F	-0.092	-0.26	-0.126	-0.153	-0.149	-0.151	-0.272	-0.191	-0.059	-0.109	-0.033	-0.019	0.139	0.06	0.054	-0.076	0.039	0.128	0.15	0.024	0.5	-1
I	-0.181	-0.126	-0.333	-0.188	-0.292	0.007	-0.266	-0.172	-0.027	-0.211	-0.05	0.046	0.319	0.035	0.009	-0.025	0.043	0.049	0.123	-0.014	0.5	-1
M	-0.116	-0.153	-0.188	-0.617	-0.206	-0.124	-0.371	-0.177	-0.079	-0.207	-0.075	-0.034	0.179	0.007	-0.041	0	0.08	0.036	0.136	0.121		0.5	-1
V	-0.214	-0.149	-0.292	-0.206	-0.404	-0.135	-0.239	-0.153	-0.01	-0.31	-0.062	-0.036	0.197	-0.035	0.043	0.021	0.139	0.096	0.148	0.057	0.5	-1
W	-0.051	-0.151	0.007	-0.124	-0.135	-0.416	-0.26	-0.253	-0.132	-0.056	0.017	-0.034	0.11	-0.111	-0.053	-0.004	0.028	0.058	0.167	0.017	0.5	-1
C	-0.164	-0.272	-0.266	-0.371	-0.239	-0.26	-2.066	-0.315	-0.192	-0.187	-0.108	-0.207	0.018	0.001	-0.036	-0.191	-0.038	0.202	0.097	0.156	0.5	-1
Y	-0.142	-0.191	-0.172	-0.177	-0.153	-0.253	-0.315	-0.238	-0.062	-0.117	-0.027	0.019	0.121	-0.06	-0.003	-0.023	-0.003	0.145	0.165	-0.013	0.5	-1
H	-0.012	-0.059	-0.027	-0.079	-0.01	-0.132	-0.192	-0.062	-0.4	-0.032	-0.026	-0.006	0.192	0.038	0.121	0.032	0.133	0.232	0.083	0.373	0.5	-1
A	-0.263	-0.109	-0.211	-0.207	-0.31	-0.056	-0.187	-0.117	-0.032	-0.559	-0.05	-0.105	0.139	-0.128	-0.07	-0.031	0.046	-0.012	0.073	0.075	0.5	-1
T	0.028	-0.033	-0.05	-0.075	-0.062	0.017	-0.108	-0.027	-0.026	-0.05	-0.112	-0.027	0.227	0.131	0.032	-0.017	0.033	0.134	0.087	0.165	0.5	-1
G	0.092	-0.019	0.046	-0.034	-0.036	-0.034	-0.207	0.019	-0.006	-0.105	-0.027	-0.138	0.067	0.084	0.134	-0.013	0.048	0.24	0.137	0.179	0.5	-1
P	0.245	0.139	0.319	0.179	0.197	0.11	0.018	0.121	0.192	0.139	0.227	0.067	0.247	0.299	0.276	0.201	0.288	0.428	0.366	0.496	0.5	-1
R	-0.092	0.06	0.035	0.007	-0.035	-0.111	0.001	-0.06	0.038	-0.128	0.131	0.084	0.299	0.02	0.077	0.127	0.209	-0.157	-0.038	0.35	0.5	-1
Q	-0.077	0.054	0.009	-0.041	0.043	-0.053	-0.036	-0.003	0.121	-0.07	0.032	0.134	0.276	0.077	-0.105	0.09	0.053	0.106	0.13	0.121	0.5	-1
S	0.02	-0.076	-0.025	0	0.021	-0.004	-0.191	-0.023	0.032	-0.031	-0.017	-0.013	0.201	0.127	0.09	-0.108	0.024	0.196	0.06	0.178		0.5	-1
N	0.139	0.039	0.043	0.08	0.139	0.028	-0.038	-0.003	0.133	0.046	0.033	0.048	0.288	0.209	0.053	0.024	-0.205	0.245	0.036	0.117	0.5	-1
E	0.016	0.128	0.049	0.036	0.096	0.058	0.202	0.145	0.232	-0.012	0.134	0.24	0.428	-0.157	0.106	0.196	0.245	0.181	0.396	-0.182	0.5	-1
D	0.169	0.15	0.123	0.136	0.148	0.167	0.097	0.165	0.083	0.073	0.087	0.137	0.366	-0.038	0.13	0.06	0.036	0.396	0.301	-0.006	0.5	-1
K	0.043	0.024	-0.014	0.121	0.057	0.017	0.156	-0.013	0.373	0.075	0.165	0.179	0.496	0.35	0.121	0.178	0.117	-0.182	-0.006	0.205	0.5	-1                                                       
X	0.5		0.5		0.5		0.5		0.5		0.5		0.5		0.5		0.5		0.5		0.5		0.5		0.5		0.5		0.5		0.5		0.5		0.5		0.5		0.5		0.5	-1
*	-1		-1		-1		-1		-1		-1		-1		-1		-1		-1		-1		-1		-1		-1		-1		-1		-1		-1		-1		-1		-1	-1
"""), sep='\s+').loc[list(aa_dict_one_hot.keys()), list(aa_dict_one_hot.keys())]
assert (Caphla_MATRIX == Caphla_MATRIX.T).all().all()

BLOSUM50_MATRIX = pd.read_table(StringIO(u"""                                                                                      
   A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V  B  J  Z  X  *                                                           
A  5 -2 -1 -2 -1 -1 -1  0 -2 -1 -2 -1 -1 -3 -1  1  0 -3 -2  0 -2 -2 -1 -1 -5                                                           
R -2  7 -1 -2 -4  1  0 -3  0 -4 -3  3 -2 -3 -3 -1 -1 -3 -1 -3 -1 -3  0 -1 -5                                                           
N -1 -1  7  2 -2  0  0  0  1 -3 -4  0 -2 -4 -2  1  0 -4 -2 -3  5 -4  0 -1 -5                                                           
D -2 -2  2  8 -4  0  2 -1 -1 -4 -4 -1 -4 -5 -1  0 -1 -5 -3 -4  6 -4  1 -1 -5                                                           
C -1 -4 -2 -4 13 -3 -3 -3 -3 -2 -2 -3 -2 -2 -4 -1 -1 -5 -3 -1 -3 -2 -3 -1 -5                                                           
Q -1  1  0  0 -3  7  2 -2  1 -3 -2  2  0 -4 -1  0 -1 -1 -1 -3  0 -3  4 -1 -5                                                           
E -1  0  0  2 -3  2  6 -3  0 -4 -3  1 -2 -3 -1 -1 -1 -3 -2 -3  1 -3  5 -1 -5                                                           
G  0 -3  0 -1 -3 -2 -3  8 -2 -4 -4 -2 -3 -4 -2  0 -2 -3 -3 -4 -1 -4 -2 -1 -5                                                           
H -2  0  1 -1 -3  1  0 -2 10 -4 -3  0 -1 -1 -2 -1 -2 -3  2 -4  0 -3  0 -1 -5                                                          
I -1 -4 -3 -4 -2 -3 -4 -4 -4  5  2 -3  2  0 -3 -3 -1 -3 -1  4 -4  4 -3 -1 -5                                                           
L -2 -3 -4 -4 -2 -2 -3 -4 -3  2  5 -3  3  1 -4 -3 -1 -2 -1  1 -4  4 -3 -1 -5                                                           
K -1  3  0 -1 -3  2  1 -2  0 -3 -3  6 -2 -4 -1  0 -1 -3 -2 -3  0 -3  1 -1 -5                                                           
M -1 -2 -2 -4 -2  0 -2 -3 -1  2  3 -2  7  0 -3 -2 -1 -1  0  1 -3  2 -1 -1 -5                                                           
F -3 -3 -4 -5 -2 -4 -3 -4 -1  0  1 -4  0  8 -4 -3 -2  1  4 -1 -4  1 -4 -1 -5                                                           
P -1 -3 -2 -1 -4 -1 -1 -2 -2 -3 -4 -1 -3 -4 10 -1 -1 -4 -3 -3 -2 -3 -1 -1 -5                                                           
S  1 -1  1  0 -1  0 -1  0 -1 -3 -3  0 -2 -3 -1  5  2 -4 -2 -2  0 -3  0 -1 -5                                                           
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  2  5 -3 -2  0  0 -1 -1 -1 -5                                                           
W -3 -3 -4 -5 -5 -1 -3 -3 -3 -3 -2 -3 -1  1 -4 -4 -3 15  2 -3 -5 -2 -2 -1 -5                                                           
Y -2 -1 -2 -3 -3 -1 -2 -3  2 -1 -1 -2  0  4 -3 -2 -2  2  8 -1 -3 -1 -2 -1 -5                                                           
V  0 -3 -3 -4 -1 -3 -3 -4 -4  4  1 -3  1 -1 -3 -2  0 -3 -1  5 -3  2 -3 -1 -5                                                           
B -2 -1  5  6 -3  0  1 -1  0 -4 -4  0 -3 -4 -2  0  0 -5 -3 -3  6 -4  1 -1 -5                                                           
J -2 -3 -4 -4 -2 -3 -3 -4 -3  4  4 -3  2  1 -3 -3 -1 -2 -1  2 -4  4 -3 -1 -5                                                           
Z -1  0  0  1 -3  4  5 -2  0 -3 -3  1 -1 -4 -1  0 -1 -2 -2 -3  1 -3  5 -1 -5                                                           
X -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -5                                                           
* -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5  1                                                           
"""), sep='\s+').loc[list(aa_dict_one_hot.keys()), list(aa_dict_one_hot.keys())]
assert (BLOSUM50_MATRIX == BLOSUM50_MATRIX.T).all().all()


ENCODING_DATA_FRAMES = {
    "Caphla": Caphla_MATRIX,
    "BLOSUM50": BLOSUM50_MATRIX,
    "one-hot": pd.DataFrame([
        [1 if i == j else 0 for i in range(len(aa_dict_one_hot.keys()))]
        for j in range(len(aa_dict_one_hot.keys()))
    ], index=aa_dict_one_hot.keys(), columns=aa_dict_one_hot.keys())
}


def hla_encode(HLA_name, encoding_method):
    '''Convert the HLAs of a sample(s) to a zero-padded (for homozygotes)
    numeric representation.

    Parameters
    ----------
        HLA_name: the name of the HLA
        encoding_method:'Caphla' or 'BLOSUM50' or 'one-hot'
    '''
    if HLA_name not in hla_dict.keys():
        if len([hla_allele for hla_allele in hla_dict.keys() if hla_allele.startswith(str(HLA_name))]) == 0:
            print('cannot find' + HLA_name)
        HLA_name = [hla_allele for hla_allele in hla_dict.keys() if hla_allele.startswith(str(HLA_name))][0]
    if HLA_name not in hla_dict.keys():
        print('Not proper HLA allele:' + HLA_name)
    HLA_sequence = hla_dict[HLA_name]
    HLA_int = [aa_dict_one_hot[char] for char in HLA_sequence]
    while len(HLA_int) != 34:
        # if the pseudo sequence length is not 34, use X for padding
        HLA_int.append(20)
    result = ENCODING_DATA_FRAMES[encoding_method].iloc[HLA_int]
    # Get a numpy array of 34 rows and 21 columns
    return np.asarray(result)


def HLAMap(dataset, encoding_method):
    '''Input a list of HLA and get a three dimentional array'''
    m = 0
    for each_HLA in dataset:
        if m == 0:
            HLA_array = hla_encode(each_HLA, encoding_method).reshape(1, 34, 21)
        else:
            HLA_array = np.append(HLA_array, hla_encode(each_HLA, encoding_method).reshape(1, 34, 21), axis=0)
        m = m + 1
    print('HLAMap done!')
    return HLA_array


def peptide_encode_HLA(peptide, maxlen, encoding_method):
    '''Convert peptide amino acid sequence to one-hot encoding,
    optionally left padded with zeros to maxlen(15).

    The letter 'X' is interpreted as the padding character and
    is assigned a value of zero.

    e.g. encode('SIINFEKL', maxlen=12)
             := [16,  8,  8, 12,  0,  0,  0,  0,  5,  4,  9, 10]

    Parameters
    ----------
    peptide:string of peptide comprising amino acids
    maxlen : int, default 15
        Pad peptides to this maximum length. If maxlen is None,
        maxlen is set to the length of the first peptide.

    Returns
    -------
    '''
    if len(peptide) > maxlen:
        msg = 'Peptide %s has length %d > maxlen = %d.'
        raise ValueError(msg % (peptide, len(peptide), maxlen))
    peptide = peptide.replace(u'\xa0', u'')  # remove non-breaking space
    o = list(map(lambda x: aa_dict_one_hot[x.upper()] if x.upper() in aa_dict_one_hot.keys() else 20, peptide))
    # if the amino acid is not valid, replace it with padding aa 'X':20
    k = len(o)
    # use 'X'(20) for padding
    o = o[:k // 2] + [20] * (int(maxlen) - k) + o[k // 2:]
    if len(o) != maxlen:
        msg = 'Peptide %s has length %d < maxlen = %d, but pad is "none".'
        raise ValueError(msg % (peptide, len(peptide), maxlen))
    result = ENCODING_DATA_FRAMES[encoding_method].iloc[o]
    return np.asarray(result)


def antigenMap(dataset, maxlen, encoding_method):
    '''Input a list of antigens and get a three dimentional array'''
    m = 0
    for each_antigen in dataset:
        if m == 0:
            antigen_array = peptide_encode_HLA(each_antigen, maxlen, encoding_method).reshape(1, maxlen, 21)
        else:
            antigen_array = np.append(antigen_array,
                                      peptide_encode_HLA(each_antigen, maxlen, encoding_method).reshape(1, maxlen, 21),
                                      axis=0)
        m = m + 1
    print('antigenMap done!')
    return antigen_array

train_antigen_list, train_HLA_list, train_label_list = preprocess(Train)

train_antigen_array = antigenMap(train_antigen_list, 25, 'BLOSUM50')
train_HLA_array = HLAMap(train_HLA_list,'BLOSUM50')

print('Map Done')

np.save('./datasets/train_HLA.npy', train_HLA_array)
np.save('./datasets/train_antigen.npy', train_antigen_array)
train_label_list = np.asarray(train_label_list)
np.save('./datasets/train_label.npy')

print('Save Done')

