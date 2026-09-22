protein_dir=<your_protein_dir>
python script/data_prep/get_substitutions.py \
    --fasta_dir data/$protein_dir/aa_seq \
    --output_dir data/$protein_dir/substitutions
