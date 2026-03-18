# SPIMI Indexer for IN6226 Assignment1

import os
import re
import heapq
import argparse
import shutil
import time
from collections import defaultdict

# Optional stemming (NLTK)
try:
    from nltk.stem import PorterStemmer
    STEMMER = PorterStemmer()
    STEMMING = True
except Exception:
    STEMMING = False
    STEMMER = None


# Tokenization + Normalisation
punct_pattern = re.compile(r"[^\w\s]")
digit_pattern = re.compile(r"\d")

def process_token(token: str) -> str | None:
    """
    Normalisation pipeline:
    1) remove punctuation
    2) lowercase
    3) remove digits
    4) length filter
    5) optional stemming
    """
    token = punct_pattern.sub("", token).lower()
    token = digit_pattern.sub("", token)  # remove digits

    if len(token) < 2:
        return None

    if STEMMING and STEMMER is not None:
        try:
            token = STEMMER.stem(token)
        except Exception:
            # fallback to original token
            pass

    return token if token else None


def token_stream(input_dir: str):
    """
    Generator that yields (term, docid).
    - Tokenisation: split on whitespace ONLY (as required).
    - docid: relative path to avoid collisions across folders.
    """
    for root, _, files in os.walk(input_dir):
        for file in files:
            if not file.endswith(".txt"):
                continue

            path = os.path.join(root, file)
            docid = os.path.relpath(path, input_dir)

            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    # split on whitespace characters
                    for word in f.read().split():
                        term = process_token(word)
                        if term:
                            yield term, docid
            except Exception as e:
                print(f"[WARN] Skip unreadable file: {path} ({e})")


# Write block to disk
def write_block(index: dict, block_id: int, temp_dir: str) -> str:
    """
    Write one SPIMI block to disk.
    Output format per line: term<TAB>docid1,docid2,...
    """
    path = os.path.join(temp_dir, f"block_{block_id}.txt")

    with open(path, "w", encoding="utf-8") as f:
        for term in sorted(index.keys()):
            postings = sorted(set(index[term]))  # dedup & sort
            f.write(f"{term}\t{','.join(postings)}\n")

    return path


# SPIMI Index Construction
def build_blocks(stream, block_size: int, temp_dir: str):
    """
    Build SPIMI blocks with a memory budget approximation controlled by block_size.
    We ensure we do not keep growing in-memory structures beyond the configured threshold.
    """
    os.makedirs(temp_dir, exist_ok=True)

    index = defaultdict(list)
    blocks = []

    block_id = 0
    approx_size = 0
    tokens_seen = 0

    t0 = time.time()

    for term, docid in stream:
        tokens_seen += 1

        index[term].append(docid)

        # Approximate memory growth (simple & stable):
        # term bytes + docid bytes + small overhead per pair
        approx_size += len(term) + len(docid) + 8

        if approx_size >= block_size:
            block_path = write_block(index, block_id, temp_dir)
            blocks.append(block_path)

            print(f"[BLOCK] {block_id} written | terms={len(index):,} | approx={approx_size/(1024*1024):.2f} MB")

            index = defaultdict(list)
            approx_size = 0
            block_id += 1

    if index:
        block_path = write_block(index, block_id, temp_dir)
        blocks.append(block_path)
        print(f"[BLOCK] {block_id} written | terms={len(index):,} | approx={approx_size/(1024*1024):.2f} MB")

    t_build = time.time() - t0
    return blocks, tokens_seen, t_build


# Merge blocks (k-way merge)
def merge_blocks(blocks: list[str], output_file: str) -> tuple[int, float]:
    """
    K-way merge of block files into one final sorted index.
    Returns: (unique_term_count, merge_time_seconds)
    """
    t0 = time.time()

    files = [open(b, "r", encoding="utf-8") for b in blocks]
    heap = []

    for i, f in enumerate(files):
        line = f.readline()
        if line:
            term = line.split("\t", 1)[0]
            heapq.heappush(heap, (term, i, line))

    term_count = 0

    with open(output_file, "w", encoding="utf-8") as out:
        current_term = None
        postings = []

        while heap:
            term, file_id, line = heapq.heappop(heap)

            parts = line.strip().split("\t", 1)
            docs = parts[1].split(",") if len(parts) == 2 and parts[1] else []

            if term == current_term:
                postings.extend(docs)
            else:
                if current_term is not None:
                    unique = sorted(set(postings))
                    out.write(f"{current_term}\t{','.join(unique)}\n")
                    term_count += 1

                current_term = term
                postings = docs

            next_line = files[file_id].readline()
            if next_line:
                next_term = next_line.split("\t", 1)[0]
                heapq.heappush(heap, (next_term, file_id, next_line))

        if current_term is not None:
            unique = sorted(set(postings))
            out.write(f"{current_term}\t{','.join(unique)}\n")
            term_count += 1

    for f in files:
        f.close()

    t_merge = time.time() - t0
    return term_count, t_merge


# Main
def main():
    parser = argparse.ArgumentParser(description="SPIMI Indexer (IN6226 Assignment 1)")
    parser.add_argument("--input-dir", required=True, help="Directory containing .txt files")
    parser.add_argument("--output", default="index.txt", help="Final index output file")
    parser.add_argument("--block-size", type=int, default=5_000_000,
                        help="Approx memory budget per block (bytes). Default=5,000,000 (~4.8MB)")
    parser.add_argument("--temp-dir", default="blocks", help="Temporary blocks directory")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        raise SystemExit(f"[ERROR] Input directory not found: {args.input_dir}")

    print("==============================================")
    print("SPIMI Indexer")
    print("==============================================")
    print(f"Input dir   : {args.input_dir}")
    print(f"Output file : {args.output}")
    print(f"Temp dir    : {args.temp_dir}")
    print(f"Block size  : {args.block_size} bytes (~{args.block_size/(1024*1024):.2f} MB)")
    print(f"Stemming    : {'ON' if STEMMING else 'OFF'}")
    print("==============================================\n")

    total_start = time.time()

    print("[1/2] Building blocks...")
    stream = token_stream(args.input_dir)
    blocks, tokens_seen, build_time = build_blocks(stream, args.block_size, args.temp_dir)
    print(f"\n[OK] {len(blocks)} blocks created")
    print(f"[STATS] Tokens processed: {tokens_seen:,}")
    print(f"[TIME] Block build time: {build_time:.2f}s\n")

    print("[2/2] Merging blocks...")
    term_count, merge_time = merge_blocks(blocks, args.output)
    print(f"\n[OK] Merge finished")
    print(f"[STATS] Unique terms (output lines): {term_count:,}")
    print(f"[TIME] Merge time: {merge_time:.2f}s\n")

    # Cleanup
    if os.path.exists(args.temp_dir):
        shutil.rmtree(args.temp_dir)

    total_time = time.time() - total_start
    out_size_mb = os.path.getsize(args.output) / (1024 * 1024)

    print("==============================================")
    print("Index Statistics")
    print("==============================================")
    print(f"Output file         : {args.output}")
    print(f"Output size         : {out_size_mb:.2f} MB")
    print(f"Tokens processed    : {tokens_seen:,}")
    print(f"Unique terms        : {term_count:,}")
    print(f"Blocks created      : {len(blocks)}")
    print(f"Build blocks time   : {build_time:.2f} s")
    print(f"Merge time          : {merge_time:.2f} s")
    print(f"Total time          : {total_time:.2f} s")
    print("==============================================")


if __name__ == "__main__":
    main()