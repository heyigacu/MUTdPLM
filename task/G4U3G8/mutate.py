
from Bio import SeqIO
import csv
import os
parent_dir = os.path.abspath(os.path.dirname(__file__))
parent_parent_dir =  os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
positions = {190: 'I', 108: 'K', 36: 'M', 96:'M', 104:'Q', 189:'S', 116:'Y'} 
output_file = f"{parent_dir}/G4U3G8_processed.csv"


records = list(SeqIO.parse(f"{parent_dir}/WT.fasta", "fasta"))
wild_seq = str(records[0].seq)

rows = []
rows.append(['UniprotID', 'WTSequence', 'MutSequence', 'Mutation', 'Label'])
for pos, wt_aa in positions.items():
    for aa in amino_acids:
        if aa != wt_aa:
            mutated = list(wild_seq)
            mutated[pos-1] = aa
            mut_name = f"{wt_aa}{pos}{aa}"
            rows.append(['G4U3G8', wild_seq, ''.join(mutated), mut_name, '0.5'])

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerows(rows)
