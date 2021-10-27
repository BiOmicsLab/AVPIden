from Bio.SeqUtils.ProtParam import ProteinAnalysis as PA
from modlamp.descriptors import PeptideDescriptor, GlobalDescriptor
from sklearn.model_selection import train_test_split
from utils import read_fasta
import pandas as pd
import os, re, math, platform
from tqdm import tqdm
from collections import Counter


# Miscellaneous
_AALetter = ['A', 'C', 'D', 'E', 'F', 'G', 'H',
             'I', 'K', 'L', 'M', 'N', 'P', 'Q',
             'R', 'S', 'T', 'V', 'W', 'Y']


"""
n_gram statistics
"""

_AALetter = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 
            'I', 'K', 'L', 'M', 'N', 'P', 'Q', 
            'R', 'S', 'T', 'V', 'W', 'Y']


def get_aan_corpus(n=2):
    '''
    Get AA corpus of n_gram (e.g. Di, Tri, etc.)
    Output: AA n_gram corpus ((e.g. Di:400, Tri:3000, etc.))
    '''
    n_corpus = []
    if n <= 2:
        for i in _AALetter:
            for j in _AALetter:
               n_corpus.append("{}{}".format(i, j))
        return n_corpus
    for i in get_aan_corpus(n - 1):
        for j in _AALetter:
            n_corpus.append("{}{}".format(i, j))
    return n_corpus


def get_ngram_counts(seq, n=2):
    '''
    Get n_gram statistics
    Input: peptide sequence and n
    Ouput: n_gram statistic (dictionary) {A.A Corp: Counts}
    '''
    # Get the name of ngram feature
    if n == 2:
        prefix = 'DPC'
    elif n == 3:
        prefix = 'TPC'
    else:
        prefix = '{}gram'.format(n)

    ngrams = [seq[i: i + n] for i in range(len(seq) - n + 1)]
    n_corpus = get_aan_corpus(n)
    ngram_stat = {}
    for aa_ng in n_corpus:
        ngram_stat['{}|{}'.format(prefix, aa_ng)] = ngrams.count(aa_ng) / len(ngrams) * 100
    return ngram_stat


def minSequenceLength(fastas):
    minLen = 10000
    for i in fastas:
        if minLen > len(i[1]):
            minLen = len(i[1])
    return minLen


def minSequenceLengthWithNormalAA(fastas):
    minLen = 10000
    for i in fastas:
        if minLen > len(re.sub('-', '', i[1])):
            minLen = len(re.sub('-', '', i[1]))
    return minLen


"""
    input.fasta:      the input protein sequence file in fasta format.
    k_space:          the gap of two amino acids, integer, defaule: 5
    output:           the encoding file, default: 'encodings.tsv'
"""


def generateGroupPairs(groupKey):
    gPair = {}
    for key1 in groupKey:
        for key2 in groupKey:
            gPair['CKSAAGP|'+key1+'.'+key2] = 0
    return gPair


def cksaagp(fastas, gap = 5, **kw):
    if gap < 0:
        print('Error: the gap should be equal or greater than zero' + '\n\n')
        return 0

    if minSequenceLength(fastas) < gap+2:
        print('Error: all the sequence length should be greater than the (gap value) + 2 = ' + str(gap+2) + '\n\n')
        return 0

    group = {'aliphatic': 'GAVLMI',
             'aromatic': 'FYW',
             'postivecharge': 'KRH',
             'negativecharge': 'DE',
             'uncharge': 'STCPNQ'}

    AA = 'ARNDCQEGHILKMFPSTWYV'

    groupKey = group.keys()

    index = {}
    for key in groupKey:
        for aa in group[key]:
            index[aa] = key

    gPairIndex = []
    for key1 in groupKey:
        for key2 in groupKey:
            gPairIndex.append('CKSAAGP|'+key1+'.'+key2)

    encodings = []
    header = ['#']
    for g in range(gap + 1):
        for p in gPairIndex:
            header.append(p+'.gap'+str(g))
    encodings.append(header)

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = [name]
        for g in range(gap + 1):
            gPair = generateGroupPairs(groupKey)
            sum = 0
            for p1 in range(len(sequence)):
                p2 = p1 + g + 1
                if p2 < len(sequence) and sequence[p1] in AA and sequence[p2] in AA:
                    gPair['CKSAAGP|'+index[sequence[p1]]+'.'+index[sequence[p2]]] = gPair['CKSAAGP|'+index[sequence[p1]]+'.'+index[sequence[p2]]] + 1
                    sum = sum + 1

            if sum == 0:
                for gp in gPairIndex:
                    code.append(0)
            else:
                for gp in gPairIndex:
                    code.append(gPair[gp] / sum)

        encodings.append(code)

    return encodings


"""
    input.fasta:      the input protein sequence file in fasta format.
    lambda:           the lambda value, integer, defaule: 30
    output:           the encoding file, default: 'encodings.tsv'
"""

def Rvalue(aa1, aa2, AADict, Matrix):
    return sum([(Matrix[i][AADict[aa1]] - Matrix[i][AADict[aa2]]) ** 2 for i in range(len(Matrix))]) / len(Matrix)


def paac(fastas, lambdaValue=30, w=0.05, **kw):
    if minSequenceLengthWithNormalAA(fastas) < lambdaValue + 1:
        print('Error: all the sequence length should be larger than the lambdaValue+1: ' + str(lambdaValue + 1) + '\n\n')
        return 0

    dataFile = re.sub('codes$', '', os.path.split(os.path.realpath(__file__))[0]) + r'\data\PAAC.txt' if platform.system() == 'Windows' else re.sub('codes$', '', os.path.split(os.path.realpath(__file__))[0]) + '/data/PAAC.txt'
    with open(dataFile) as f:
        records = f.readlines()
    AA = ''.join(records[0].rstrip().split()[1:])
    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i
    AAProperty = []
    AAPropertyNames = []
    for i in range(1, len(records)):
        array = records[i].rstrip().split() if records[i].rstrip() != '' else None
        AAProperty.append([float(j) for j in array[1:]])
        AAPropertyNames.append(array[0])

    AAProperty1 = []
    for i in AAProperty:
        meanI = sum(i) / 20
        fenmu = math.sqrt(sum([(j-meanI)**2 for j in i])/20)
        AAProperty1.append([(j-meanI)/fenmu for j in i])

    encodings = []
    header = ['#']
    for aa in AA:
        header.append('PAAC|' + aa)
    for n in range(1, lambdaValue + 1):
        header.append('PAAC|lambda' + str(n))
    encodings.append(header)

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = [name]
        theta = []
        for n in range(1, lambdaValue + 1):
            theta.append(
                sum([Rvalue(sequence[j], sequence[j + n], AADict, AAProperty1) for j in range(len(sequence) - n)]) / (
                len(sequence) - n))
        myDict = {}
        for aa in AA:
            myDict[aa] = sequence.count(aa)
        code = code + [myDict[aa] / (1 + w * sum(theta)) for aa in AA]
        code = code + [(w * j) / (1 + w * sum(theta)) for j in theta]
        encodings.append(code)
    return encodings


def GAAC(fastas, **kw):
	group = {
		'alphatic': 'GAVLMI',
		'aromatic': 'FYW',
		'postivecharge': 'KRH',
		'negativecharge': 'DE',
		'uncharge': 'STCPNQ'
	}

	groupKey = group.keys()

	encodings = []
	header = ['#']
	for key in groupKey:
		header.append("GAAC|"+key)
	encodings.append(header)

	for i in fastas:
		name, sequence = i[0], re.sub('-', '', i[1])
		code = [name]
		count = Counter(sequence)
		myDict = {}
		for key in groupKey:
			for aa in group[key]:
				myDict[key] = myDict.get(key, 0) + count[aa]

		for key in groupKey:
			code.append(myDict[key]/len(sequence))
		encodings.append(code)

	return encodings


def GDPC(fastas, **kw):
	group = {
		'alphaticr': 'GAVLMI',
		'aromatic': 'FYW',
		'postivecharger': 'KRH',
		'negativecharger': 'DE',
		'uncharger': 'STCPNQ'
	}

	groupKey = group.keys()
	baseNum = len(groupKey)
	dipeptide = [g1+'.'+g2 for g1 in groupKey for g2 in groupKey]

	index = {}
	for key in groupKey:
		for aa in group[key]:
			index[aa] = key

	encodings = []
	header = ['#'] + ['GDPC|'+dipname for dipname in dipeptide]
	encodings.append(header)

	for i in fastas:
		name, sequence = i[0], re.sub('-', '', i[1])

		code = [name]
		myDict = {}
		for t in dipeptide:
			myDict[t] = 0

		sum = 0
		for j in range(len(sequence) - 2 + 1):
			myDict[index[sequence[j]]+'.'+index[sequence[j+1]]] = myDict[index[sequence[j]]+'.'+index[sequence[j+1]]] + 1
			sum = sum +1

		if sum == 0:
			for t in dipeptide:
				code.append(0)
		else:
			for t in dipeptide:
				code.append(myDict[t]/sum)
		encodings.append(code)

	return encodings


def GTPC(fastas, **kw):
	group = {
		'alphaticr': 'GAVLMI',
		'aromatic': 'FYW',
		'postivecharger': 'KRH',
		'negativecharger': 'DE',
		'uncharger': 'STCPNQ'
	}

	groupKey = group.keys()
	baseNum = len(groupKey)
	triple = [g1+'.'+g2+'.'+g3 for g1 in groupKey for g2 in groupKey for g3 in groupKey]

	index = {}
	for key in groupKey:
		for aa in group[key]:
			index[aa] = key

	encodings = []
	header = ['#'] + ['GTPC|'+tname for tname in triple]
	encodings.append(header)

	for i in fastas:
		name, sequence = i[0], re.sub('-', '', i[1])

		code = [name]
		myDict = {}
		for t in triple:
			myDict[t] = 0

		sum = 0
		for j in range(len(sequence) - 3 + 1):
			myDict[index[sequence[j]]+'.'+index[sequence[j+1]]+'.'+index[sequence[j+2]]] = myDict[index[sequence[j]]+'.'+index[sequence[j+1]]+'.'+index[sequence[j+2]]] + 1
			sum = sum +1

		if sum == 0:
			for t in triple:
				code.append(0)
		else:
			for t in triple:
				code.append(myDict[t]/sum)
		encodings.append(code)

	return encodings

'''
Insert Iso_electric Point and net_charge(neutral) feature to the sequence data_frame
Input: sequence data_frame {IDx: Seq_x, ...}
Output: data_frame of Peptide Seq {IDx: Seq_x, ..., iep, net_charge}
'''


def insert_phycs(seq_df):
    seq_df = seq_df.copy()
    #  Function for compute Isoelectric Point or net_charge of peptide
    def get_ieq_nc(seq, is_iep=True):
        protparam = PA(seq)
        return protparam.isoelectric_point() if is_iep else protparam.charge_at_pH(7.0)

    # Calculating IsoElectricPoints and NeutralCharge
    data_size = seq_df.size
    seq_df['PHYC|IEP'] = list(map(get_ieq_nc, seq_df['Sequence'], [True] * data_size))  # IsoElectricPoints
    seq_df['PHYC|Net Charge'] = list(map(get_ieq_nc, seq_df['Sequence'], [False] * data_size))  # Charge(Neutral)

    # Calculating hydrophobic moment (My assume all peptides are alpha-helix)
    descrpt = PeptideDescriptor(seq_df['Sequence'].values, 'eisenberg')
    descrpt.calculate_moment(window=1000, angle=100, modality='max')
    seq_df['PHYC|Hydrophobic Moment'] = descrpt.descriptor.reshape(-1)

    # Calculating "Hopp-Woods" hydrophobicity
    descrpt = PeptideDescriptor(seq_df['Sequence'].values, 'hopp-woods')
    descrpt.calculate_global()
    seq_df['PHYC|Hydrophobicity'] = descrpt.descriptor.reshape(-1)

    # Calculating Energy of Transmembrane Propensity
    descrpt = PeptideDescriptor(seq_df['Sequence'].values, 'tm_tend')
    descrpt.calculate_global()
    seq_df['PHYC|Transmembrane Propensity'] = descrpt.descriptor.reshape(-1)

    # Calculating Aromaticity
    descrpt = GlobalDescriptor(seq_df['Sequence'].values)
    descrpt.aromaticity()
    seq_df['PHYC|Aromacity'] = descrpt.descriptor.reshape(-1)

    # Calculating Levitt_alpha_helical Propensity
    descrpt = PeptideDescriptor(seq_df['Sequence'].values, 'levitt_alpha')
    descrpt.calculate_global()
    seq_df['PHYC|Alpha Helical Propensity'] = descrpt.descriptor.reshape(-1)

    # Calculating Aliphatic Index
    descrpt = GlobalDescriptor(seq_df['Sequence'].values)
    descrpt.aliphatic_index()
    seq_df['PHYC|Aliphatic Index'] = descrpt.descriptor.reshape(-1)

    # Calculating Boman Index
    descrpt = GlobalDescriptor(seq_df['Sequence'].values)
    descrpt.boman_index()
    seq_df['PHYC|Boman Index'] = descrpt.descriptor.reshape(-1)

    return seq_df


'''
Insert Amino acid composition to the sequence data_frame
Input: sequence data_frame {IDx: Seq_x}
Output: data_frame of Peptide Seq {IDx: Seq_x, ..., AAC_Ax ... AAC_Yx}
'''


def insert_aac(seq_df):
    seq_df = seq_df.copy()
    # Compute AAC for peptide in specific A.A
    def get_aac(seq, aa):
        return seq.count(aa) / len(seq) * 100

    # processing data_frame
    data_size = seq_df.size
    for ll in _AALetter:
        seq_df['AAC|{}'.format(ll)] = list(map(get_aac, seq_df['Sequence'], [ll] * data_size))
    return seq_df


'''
Insert n_grams Descriptor to the sequence data_frame
Input: sequence data_frame {IDx: Seq_x, ...}
Output: data_frame of Peptide Seq {IDx: Seq_x, ..., ngram_(1), .., ngram(20 ** n)}
'''


def insert_ngrams(seq_df, n=2):
    seq_df = seq_df.copy()
    data_size = seq_df.size

    ngrams_df = list(map(get_ngram_counts, seq_df['Sequence'], [n] * data_size))
    ngrams_df = pd.DataFrame(ngrams_df)  # Convert ngrams features to pd.DataFrame
    seq_df = pd.concat([seq_df, ngrams_df], axis=1)
    return seq_df


"""
Insert CKSAAGP Descriptor to the sequence data_frame
(Composition of k-spaced amino acid pairs)
Input: sequence data_frame {IDx: Seq_x, ...}
Output: data_frame of Peptide Seq {IDx: Seq_x, ..., CKSAAGP(0), ..., CKSAAGP(m)}
"""


def insert_cksaagp(seq_df, gap=2):
    seq_df.copy()
    fastas = [[idx, seq] for idx, seq in zip(seq_df['Id'], seq_df['Sequence'])]
    encoding = cksaagp(fastas, gap=gap)
    encoding = pd.DataFrame(encoding[1:], columns=encoding[0])
    seq_df = pd.concat([seq_df, encoding.iloc[:, 1:]], axis=1)
    return seq_df


"""
Insert PAAC Descriptor to the sequence data_frame
(Pseudo Amino Acid Composition)
Input: sequence data_frame {IDx: Seq_x, ...}
Output: data_frame of Peptide Seq {IDx: Seq_x, ..., CKSAAGP(0), ..., CKSAAGP(m)}
"""


def insert_paac(seq_df, lamb=3, w=0.1):
    seq_df = seq_df.copy()
    fastas = [[idx, seq] for idx, seq in zip(seq_df['Id'], seq_df['Sequence'])]
    encoding = paac(fastas, lambdaValue=lamb, w=w)
    encoding = pd.DataFrame(encoding[1:], columns=encoding[0])
    seq_df = pd.concat([seq_df, encoding.iloc[:, 1:]], axis=1)
    return seq_df


"""
Insert Grouped n-gram Descriptor to the sequence data_frame
(Pseudo Amino Acid Composition)
Input: sequence data_frame {IDx: Seq_x, ...}
Output: data_frame of Peptide Seq {IDx: Seq_x, ..., CKSAAGP(0), ..., CKSAAGP(m)}
"""


def insert_gngram(seq_df, n=1):
    seq_df = seq_df.copy()
    fastas = [[idx, seq] for idx, seq in zip(seq_df['Id'], seq_df['Sequence'])]
    # encoding = paac(fastas, lambdaValue=lamb, w=w)
    if n is 1:
        encoding = GAAC(fastas)
    elif n is 2:
        encoding = GDPC(fastas)
    elif n is 3:
        encoding = GTPC(fastas)
    else:
        raise Warning("Invalid n-grams, no features added")
    encoding = pd.DataFrame(encoding[1:], columns=encoding[0])
    seq_df = pd.concat([seq_df, encoding.iloc[:, 1:]], axis=1)
    return seq_df


def construct_features(seq_df, paaclamb=6, paacw=0.5):
    """
    Construct Features for the AVPIden. We first investigated physiochemical feautres, AAC features, DiC features, 
    CKSAAGP features, PAAC features, and PHYC features.
    Parameters are pre-set accroding to the sequence identities.
    """
    seq_df = insert_aac(seq_df)
    seq_df = insert_ngrams(seq_df, n=2)
    seq_df = insert_cksaagp(seq_df, gap=3) # As the maximum motif length = 5.
    seq_df = insert_paac(seq_df, lamb=paaclamb, w=paacw)
    seq_df = insert_phycs(seq_df)

    return seq_df


if __name__ == "__main__":

    # Entire
    print("Translate the entire-stage prediction data...", flush=True)
    entire_datadir = "data/Entire/set"
    if not os.path.exists(entire_datadir):
        os.makedirs(entire_datadir)
    labels = ["Anti-Virus", "non-AMP", "non-AVP"]
    for l in tqdm(labels):
        df = read_fasta("./Fasta/Entire/fasta/{:s}.faa".format(l))
        df = construct_features(df)
        df.to_csv(os.path.join(entire_datadir, "{:s}.csv".format(l)), index=False)
    print("Done!", flush=True)
    # ByFamily
    print("Translate the By-Family prediction data...", flush=True)
    family_datadir = "data/ByFamily/set"
    if not os.path.exists(family_datadir):
        os.makedirs(family_datadir)
    families = ["Coronaviridae", "Flaviviridae", "Herpesviridae", 
    "Orthomyxoviridae", "Paramyxoviridae", "Retroviridae"]
    families = families + list(map(lambda x: "non-"+x, families))
    for fm in tqdm(families):
        df = read_fasta("./Fasta/ByFamily/fasta/{:s}.faa".format(fm))
        df = construct_features(df)
        df.to_csv(os.path.join(family_datadir, "{:s}.csv".format(fm)), index=False)
    print("Done!", flush=True)
    # By virus
    print("Translate the By-Virus prediction data...", flush=True)
    virus_datadir = "data/ByVirus/set"
    if not os.path.exists(virus_datadir):
        os.makedirs(virus_datadir)
    viruses = ["FIV", "HCV", "HIV", "HPIV3", "HSV1", "INFVA",  "RSV", "SARSCoV"]
    viruses = viruses + list(map(lambda x: "non-"+x, viruses))
    for vm in tqdm(viruses):
        df = read_fasta("./Fasta/ByVirus/fasta/{:s}.faa".format(vm))
        df = construct_features(df)
        df.to_csv(os.path.join(virus_datadir, "{:s}.csv".format(vm)), index=False)
    print("Done!", flush=True)
