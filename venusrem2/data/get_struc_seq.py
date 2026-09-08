import argparse
import os

from tqdm import tqdm

from venusrem2.baseline.prosst.get_sst_seq import SSTPredictor


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_dir", type=str, default=None, help="Directory containing PDB files",)
    parser.add_argument("--pdb_file", type=str, default=None, help="PDB file",)
    parser.add_argument("--vocab_size", type=int, default=[2048], nargs='+', help="Vocabulary size",)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files",)
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory",)
    parser.add_argument("--num_processes", type=int, default=12, help="Number of processes for parallel PDB processing",)
    parser.add_argument("--num_threads", type=int, default=16, help="Number of threads for subgraph generation",)
    parser.add_argument("--max_batch_nodes", type=int, default=10000, help="Maximum nodes per batch for GVP inference",)
    parser.add_argument("--cache_subgraph_dir", type=str, default=None, help="Directory to cache subgraphs for reuse",)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for v in args.vocab_size:
        os.makedirs(os.path.join(args.output_dir, str(v)), exist_ok=True)

    if args.cache_subgraph_dir:
        os.makedirs(args.cache_subgraph_dir, exist_ok=True)

    if args.pdb_dir is not None:
        pdb_files = sorted(os.listdir(args.pdb_dir))
        pdb_files = [os.path.join(args.pdb_dir, pdb_file) for pdb_file in pdb_files]
    elif args.pdb_file is not None:
        pdb_files = [args.pdb_file]
    else:
        raise ValueError("Either pdb_dir or pdb_file must be provided")

    for v in args.vocab_size:
        if not args.overwrite:
            out_subdir = os.path.join(args.output_dir, str(v))
            existing = {f.replace('.fasta', '') for f in os.listdir(out_subdir) if f.endswith('.fasta')}
            before = len(pdb_files)
            pdb_files_filtered = [p for p in pdb_files if os.path.basename(p).split('.')[0] not in existing]
            skipped = before - len(pdb_files_filtered)
            if skipped > 0:
                print(f"Skipping {skipped} already processed PDBs (use --overwrite to redo)")
            pdb_files_v = pdb_files_filtered
        else:
            pdb_files_v = pdb_files

        if not pdb_files_v:
            print(f"All PDBs already processed for vocab_size={v}, nothing to do")
            continue

        processor = SSTPredictor(
            structure_vocab_size=v,
            num_processes=args.num_processes,
            num_threads=args.num_threads,
            max_batch_nodes=args.max_batch_nodes,
        )
        results = processor.predict_from_pdb(
            pdb_files_v,
            cache_subgraph_dir=args.cache_subgraph_dir,
        )
        for result in results:
            name = result['name'].split('.')[0]
            sst_seq = result[f'{v}_sst_seq']
            sst_seq = [str(i) for i in sst_seq]
            with open(os.path.join(args.output_dir, str(v), name+'.fasta'), "w") as f:
                f.write(f'>{name}\n')
                f.write(','.join(sst_seq))
                