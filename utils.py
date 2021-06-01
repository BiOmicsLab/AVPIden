import pandas as pd
from Bio import Seq, SeqIO


def write_fasta(df, file_path, abbr_columns=None):
    """
    Save dataframe to a .fasta file, the df should contain at least columns named "Id" and "Sequence"
    
    df: dataframe for saving .fasta
    file_path: path(string) for saving the fasta file
    abbr_columns: string columns for adding abbreviations. Multiple abbr are splited by '|'.
    """
    Seqrecords = [SeqIO.SeqRecord(id=row['Id'], 
                              seq=Seq.Seq(row['Sequence']), 
                              description='|'.join(row[abbr_columns] if abbr_columns is not None else "")) \
             for idn, row in df.iterrows()]
    with open(file_path, 'w+') as fhandle:
        SeqIO.write(Seqrecords, fhandle, "fasta-2line")
        print("Saved {:d} sequences.".format(len(Seqrecords)))


def read_fasta(fname):
    '''
    Read fasta file to dictionary
    Input: path name of fasta
    Output: dataframe of Peptide Seq {ID1: Seq1, ID2: Seq2,...}
    '''
    with open(fname, "rU") as f:
        seq_dict = [(record.id, record.seq._data) for record in SeqIO.parse(f, "fasta")]
    seq_df = pd.DataFrame(data=seq_dict, columns=["Id", "Sequence"])
    return seq_df

