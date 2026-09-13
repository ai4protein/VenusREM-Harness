#!/bin/bash
# Batch-fold single-chain proteins via ESMFold v1 (ESM Atlas free API) and/or
# ESMFold2 (Biohub API). Default runs both — set BIOHUB_TOKEN before v2 is used.
# Usage:
#   export BIOHUB_TOKEN=<your-token>
#   bash script/structure/fold_esmfold2_api.sh <dataset> [backends]
# Examples:
#   bash script/structure/fold_esmfold2_api.sh case            # both v1 + v2
#   bash script/structure/fold_esmfold2_api.sh case v1         # v1 only (no token needed)
#   bash script/structure/fold_esmfold2_api.sh case v2         # v2 only
# Outputs:
#   data/<dataset>/esmfold_pdbs/v1/<name>.{pdb,plddt.json}
#   data/<dataset>/esmfold_pdbs/v2/<name>.{cif,pdb,plddt.json}
dataset=${1:-case}
backends=${2:-v1,v2}

python script/structure/esmfold2.py \
    --aa_seq_dir data/${dataset}/aa_seq \
    --out_dir    data/${dataset}/esmfold_pdbs \
    --backends   ${backends} \
    --workers    4
