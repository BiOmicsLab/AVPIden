# AVPIden

A prediction scheme for identification and functional characterization of antiviral peptides.

## Requirements

We have already integrate the environment in `env.yaml`. execute `conda create -f env.yaml` to install packages required in a new created `AVPIden` conda env.

Enter the enviornment with `conda activate AVPIden` before further executions.

## Frameworks for establishment of AVPIden

- `feature_extract.py`: modules and executions to extract features from the sequences loacted at `Fasta` directory. The result feature representations of all sequences are loacted at the `data` directory. You should run this before making classification.
- `Args.py`: Parameters of how to perform classification. You can set the parameter `stage` as one of the string from `['Entire', 'ByFamily', 'ByVirus']` to perform classifier establishment and evaluation of first stage indentification, second stage classificatoin (by virus families), or second stage classification (by specific virus), respectively.
- `classify.py`: Make classification for different antiviral peptide identification and functional characterization tasks. The evaluation of classification are also included.
- `utils.py` utilities of the framework
- If you want to illustrtes the features or classification results, funtions located in `feature_description.py` or `plot_utils.py` may help you.
- `Fasta` original fasta sequences collection of construction/evaluation of AVPIden.
