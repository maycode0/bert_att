import argparse, os, sys, torch, numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--embedding_path", default="counter-fitted-vectors.txt", help="Path to counter-fitted-vectors.txt")
parser.add_argument("--out_path", default="cos_sim_counter_fitting.npy", help="Output .npy path")
parser.add_argument("--block_size", type=int, default=1024, help="Rows per block for chunked matmul")
parser.add_argument("--dtype", choices=["float32", "float16"], default="float32", help="Compute dtype (float16 on CUDA reduces memory)")
parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Force device")
args = parser.parse_args()

embedding_path = args.embedding_path
if not os.path.exists(embedding_path):
    print(f"Error: {embedding_path} not found.")
    sys.exit(1)

print("Loading embeddings...")
embeddings_list = []
with open(embedding_path, 'r', encoding='utf-8') as ifile:
    for line in ifile:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        embeddings_list.append([float(num) for num in parts[1:]])

if args.device == "cpu":
    device = torch.device("cpu")
elif args.device == "cuda":
    device = torch.device("cuda")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

use_fp16 = (args.dtype == "float16" and device.type == "cuda")
dtype = torch.float16 if use_fp16 else torch.float32
embeddings = torch.tensor(embeddings_list, dtype=dtype, device=device)
norm = torch.norm(embeddings, p=2, dim=1, keepdim=True)
embeddings = embeddings / norm
print("Computing cosine similarity matrix...")
N = len(embeddings_list)
mm_dtype = np.float16 if dtype == torch.float16 else np.float32
out_path = args.out_path
block_size = args.block_size
out = np.lib.format.open_memmap(out_path, mode="w+", dtype=mm_dtype, shape=(N, N))

with torch.no_grad():
    for i in range(0, N, block_size):
        block = embeddings[i:i+block_size]
        prod = torch.matmul(block, embeddings.t())
        out[i:i+block_size, :] = prod.detach().cpu().numpy().astype(mm_dtype)
        del prod, block
        if device.type == "cuda":
            torch.cuda.empty_cache()

print(f"Saved to {out_path}")
print("Done.")