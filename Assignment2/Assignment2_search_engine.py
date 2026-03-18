import os
import re
import time
import pickle
import argparse

# Optional stemming
try:
    from nltk.stem import PorterStemmer
    STEMMER = PorterStemmer()
    STEMMING = True
except Exception:
    STEMMER = None
    STEMMING = False


# ============================================================
# 1. Normalization
# ============================================================

PUNCT_PATTERN = re.compile(r"[^\w\s]")
DIGIT_PATTERN = re.compile(r"\d")
DOCID_NUM_PATTERN = re.compile(r"(\d+)")


def normalize_token(token):
    """
    Match Assignment 1 normalization:
    1) remove punctuation
    2) lowercase
    3) remove digits
    4) length filter
    5) optional stemming
    """
    token = PUNCT_PATTERN.sub("", token).lower()
    token = DIGIT_PATTERN.sub("", token)

    if len(token) < 2:
        return None

    if STEMMING and STEMMER is not None:
        try:
            token = STEMMER.stem(token)
        except Exception:
            pass

    return token if token else None


def normalize_query(query):
    terms = []
    for raw in query.split():
        term = normalize_token(raw)
        if term:
            terms.append(term)
    return terms


# ============================================================
# 2. Load the inverted index produced in Assignment 1
# ============================================================

def load_inverted_index(index_file):
    """
    Load Assignment 1 index format:
    term<TAB>doc1,doc2,doc3,...
    """
    inverted_index = {}

    with open(index_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue

            parts = line.split("\t", 1)
            term = parts[0].strip()

            if not term:
                continue

            if len(parts) == 1 or not parts[1].strip():
                postings = []
            else:
                postings = [doc.strip() for doc in parts[1].split(",") if doc.strip()]

            inverted_index[term] = sorted(set(postings))

    return inverted_index


# ============================================================
# 3. Boolean search
# ============================================================

def and_search(query, inverted_index):
    terms = normalize_query(query)

    if not terms:
        return []

    for term in terms:
        if term not in inverted_index:
            return []

    result = set(inverted_index[terms[0]])
    for term in terms[1:]:
        result &= set(inverted_index[term])

    return sorted(result)


def or_search(query, inverted_index):
    terms = normalize_query(query)

    if not terms:
        return []

    result = set()
    for term in terms:
        result |= set(inverted_index.get(term, []))

    return sorted(result)


def not_search(term, inverted_index, universe_docs):
    terms = normalize_query(term)
    if not terms:
        return sorted(universe_docs)

    target_term = terms[0]
    postings = set(inverted_index.get(target_term, []))
    return sorted(universe_docs - postings)


def and_search_optimized(query, inverted_index):
    terms = normalize_query(query)

    if not terms:
        return []

    for term in terms:
        if term not in inverted_index:
            return []

    terms.sort(key=lambda t: len(inverted_index[t]))

    result = set(inverted_index[terms[0]])
    for term in terms[1:]:
        result &= set(inverted_index[term])
        if not result:
            break

    return sorted(result)


# ============================================================
# 4. Compression - Dictionary-as-a-String
# ============================================================

def compress_dictionary_as_string(terms):
    """
    Store all terms in one long string.
    Return:
      dictionary_string
      term_positions[term] = (start, length)
    """
    dictionary_string_parts = []
    term_positions = {}
    current_pos = 0

    for term in sorted(terms):
        dictionary_string_parts.append(term)
        term_positions[term] = (current_pos, len(term))
        current_pos += len(term)

    dictionary_string = "".join(dictionary_string_parts)
    return dictionary_string, term_positions


def recover_term(dictionary_string, start, length):
    return dictionary_string[start:start + length]


# ============================================================
# 5. Compression - docID integer conversion
# ============================================================

def extract_numeric_docid(docname):
    """
    Example:
    bbc_cnn_news_data12527.txt -> 12527
    """
    match = DOCID_NUM_PATTERN.search(docname)
    if not match:
        raise ValueError("Cannot extract numeric docID from: " + str(docname))
    return int(match.group(1))


def build_doc_mappings(inverted_index):
    """
    Build:
      doc_to_int[str] -> int
      int_to_doc[int] -> str
    """
    all_docs = sorted(set(doc for postings in inverted_index.values() for doc in postings))

    extracted_ids = []
    unique_ok = True
    seen = set()

    for doc in all_docs:
        try:
            num = extract_numeric_docid(doc)
        except ValueError:
            unique_ok = False
            break

        if num in seen:
            unique_ok = False
            break

        seen.add(num)
        extracted_ids.append((doc, num))

    if unique_ok:
        doc_to_int = dict((doc, num) for doc, num in extracted_ids)
        int_to_doc = dict((num, doc) for doc, num in extracted_ids)
    else:
        doc_to_int = dict((doc, i + 1) for i, doc in enumerate(all_docs))
        int_to_doc = dict((i + 1, doc) for i, doc in enumerate(all_docs))

    return doc_to_int, int_to_doc


# ============================================================
# 6. Compression - Gap encoding
# ============================================================

def gap_encode(postings):
    if not postings:
        return []

    postings = sorted(postings)
    gaps = [postings[0]]
    for i in range(1, len(postings)):
        gaps.append(postings[i] - postings[i - 1])
    return gaps


def gap_decode(gaps):
    if not gaps:
        return []

    postings = [gaps[0]]
    for i in range(1, len(gaps)):
        postings.append(postings[-1] + gaps[i])
    return postings


# ============================================================
# 7. Compression - Variable-byte encoding
# ============================================================

def vb_encode_number(n):
    if n < 0:
        raise ValueError("Variable-byte encoding only supports non-negative integers.")

    bytes_list = [n % 128]
    n //= 128

    while n > 0:
        bytes_list.insert(0, n % 128)
        n //= 128

    bytes_list[-1] += 128
    return bytes_list


def vb_encode_list(numbers):
    encoded = []
    for n in numbers:
        encoded.extend(vb_encode_number(n))
    return bytes(encoded)


def vb_decode(bytestream):
    numbers = []
    n = 0

    for b in bytestream:
        if b < 128:
            n = 128 * n + b
        else:
            n = 128 * n + (b - 128)
            numbers.append(n)
            n = 0

    return numbers


# ============================================================
# 8. Build compressed index
# ============================================================

def build_compressed_index(inverted_index):
    """
    Build a compressed structure containing:
    - dictionary_string
    - term_positions
    - compressed_postings (term -> variable-byte encoded gap list)
    - doc mappings
    """
    terms = sorted(inverted_index.keys())
    dictionary_string, term_positions = compress_dictionary_as_string(terms)
    doc_to_int, int_to_doc = build_doc_mappings(inverted_index)

    compressed_postings = {}

    for term, postings_docs in inverted_index.items():
        posting_ints = sorted(doc_to_int[doc] for doc in postings_docs)
        gaps = gap_encode(posting_ints)
        encoded = vb_encode_list(gaps)
        compressed_postings[term] = encoded

    return {
        "dictionary_string": dictionary_string,
        "term_positions": term_positions,
        "compressed_postings": compressed_postings,
        "doc_to_int": doc_to_int,
        "int_to_doc": int_to_doc,
    }


def decode_postings_for_term(term, compressed_index):
    compressed_postings = compressed_index["compressed_postings"]
    int_to_doc = compressed_index["int_to_doc"]

    if term not in compressed_postings:
        return []

    encoded = compressed_postings[term]
    gaps = vb_decode(encoded)
    posting_ints = gap_decode(gaps)
    return [int_to_doc[i] for i in posting_ints]


def and_search_compressed(query, compressed_index):
    terms = normalize_query(query)

    if not terms:
        return []

    compressed_postings = compressed_index["compressed_postings"]

    for term in terms:
        if term not in compressed_postings:
            return []

    decoded_lists = []
    for term in terms:
        decoded_lists.append(set(decode_postings_for_term(term, compressed_index)))

    result = decoded_lists[0]
    for s in decoded_lists[1:]:
        result &= s

    return sorted(result)


# ============================================================
# 9. Size / performance utilities
# ============================================================

def deep_sizeof_pickle(obj):
    return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))


def human_readable_bytes(n):
    if n < 1024:
        return str(n) + " B"
    elif n < 1024 ** 2:
        return "{:.2f} KB".format(n / 1024.0)
    elif n < 1024 ** 3:
        return "{:.2f} MB".format(n / float(1024 ** 2))
    return "{:.2f} GB".format(n / float(1024 ** 3))


def benchmark_queries(queries, inverted_index, compressed_index):
    rows = []

    for q in queries:
        t0 = time.perf_counter()
        original_result = and_search_optimized(q, inverted_index)
        original_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        compressed_result = and_search_compressed(q, compressed_index)
        compressed_time = time.perf_counter() - t1

        rows.append({
            "query": q,
            "original_hits": len(original_result),
            "compressed_hits": len(compressed_result),
            "same_result": original_result == compressed_result,
            "original_time_ms": original_time * 1000,
            "compressed_time_ms": compressed_time * 1000,
        })

    return rows


def print_benchmark_table(rows):
    print("\n================ Query Benchmark ================")
    print("{:<30} {:>10} {:>10} {:>8} {:>12} {:>12}".format(
        "Query", "Orig Hits", "Comp Hits", "Same?", "Orig(ms)", "Comp(ms)"
    ))
    print("-" * 92)

    for row in rows:
        q = row["query"]
        if len(q) > 30:
            q = q[:28] + ".."

        print("{:<30} {:>10} {:>10} {:>8} {:>12.3f} {:>12.3f}".format(
            q,
            row["original_hits"],
            row["compressed_hits"],
            str(row["same_result"]),
            row["original_time_ms"],
            row["compressed_time_ms"]
        ))


# ============================================================
# 10. Save / load compressed index
# ============================================================

def save_compressed_index(compressed_index, output_file):
    with open(output_file, "wb") as f:
        pickle.dump(compressed_index, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_compressed_index(input_file):
    with open(input_file, "rb") as f:
        return pickle.load(f)


# ============================================================
# 11. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Assignment 2 Search Engine")
    parser.add_argument("--index-file", required=True, help="Path to Assignment 1 inverted index file")
    parser.add_argument(
        "--mode",
        choices=["demo", "query", "compress", "benchmark"],
        default="demo",
        help="Run mode"
    )
    parser.add_argument("--query", type=str, default="", help="Query string for query mode")
    parser.add_argument("--save-compressed", type=str, default="", help="Optional path to save compressed index")
    parser.add_argument("--load-compressed", type=str, default="", help="Optional path to load compressed index")
    args = parser.parse_args()

    if not os.path.isfile(args.index_file):
        raise SystemExit("[ERROR] Index file not found: " + args.index_file)

    print("====================================================")
    print("Assignment 2 Search Engine")
    print("====================================================")
    print("Index file : " + args.index_file)
    print("Stemming   : " + ("ON" if STEMMING else "OFF"))
    print("====================================================\n")

    t0 = time.perf_counter()
    inverted_index = load_inverted_index(args.index_file)
    load_time = time.perf_counter() - t0

    total_terms = len(inverted_index)
    universe_docs = set(doc for postings in inverted_index.values() for doc in postings)
    total_docs = len(universe_docs)

    print("[OK] Loaded inverted index in {:.3f} s".format(load_time))
    print("[STATS] Unique terms     : {:,}".format(total_terms))
    print("[STATS] Unique documents : {:,}".format(total_docs))

    original_mem_est = deep_sizeof_pickle(inverted_index)
    print("[SIZE] Original index (pickle estimate): " + human_readable_bytes(original_mem_est))

    if args.load_compressed:
        if not os.path.isfile(args.load_compressed):
            raise SystemExit("[ERROR] Compressed index file not found: " + args.load_compressed)
        t1 = time.perf_counter()
        compressed_index = load_compressed_index(args.load_compressed)
        compress_time = time.perf_counter() - t1
        print("[OK] Loaded compressed index in {:.3f} s".format(compress_time))
    else:
        t1 = time.perf_counter()
        compressed_index = build_compressed_index(inverted_index)
        compress_time = time.perf_counter() - t1
        print("[OK] Built compressed index in {:.3f} s".format(compress_time))

    compressed_mem_est = deep_sizeof_pickle(compressed_index)
    print("[SIZE] Compressed index (pickle estimate): " + human_readable_bytes(compressed_mem_est))

    if original_mem_est > 0:
        saving_pct = 100.0 * (original_mem_est - compressed_mem_est) / original_mem_est
        print("[SAVE] Estimated memory reduction: {:.2f}%".format(saving_pct))

    if args.save_compressed:
        save_compressed_index(compressed_index, args.save_compressed)
        on_disk = os.path.getsize(args.save_compressed)
        print("[OK] Compressed index saved to: " + args.save_compressed)
        print("[SIZE] Saved compressed file size: " + human_readable_bytes(on_disk))

    if args.mode == "compress":
        print("\nCompression mode finished.")
        return

    if args.mode == "query":
        if not args.query.strip():
            raise SystemExit("[ERROR] Please provide --query for query mode.")

        print("\n================ Query Result ================")
        print("Raw query       : " + args.query)
        print("Normalized terms:", normalize_query(args.query))

        and_results = and_search_optimized(args.query, inverted_index)
        comp_results = and_search_compressed(args.query, compressed_index)
        or_results = or_search(args.query, inverted_index)

        print("\nAND results ({} docs):".format(len(and_results)))
        for doc in and_results[:20]:
            print("  " + doc)
        if len(and_results) > 20:
            print("  ... ({} more)".format(len(and_results) - 20))

        print("\nCompressed AND results match: {}".format(and_results == comp_results))

        print("\nOR results ({} docs):".format(len(or_results)))
        for doc in or_results[:20]:
            print("  " + doc)
        if len(or_results) > 20:
            print("  ... ({} more)".format(len(or_results) - 20))

        normalized_terms = normalize_query(args.query)
        if normalized_terms:
            not_results = not_search(normalized_terms[0], inverted_index, universe_docs)
            print("\nNOT {} results: {} docs".format(normalized_terms[0], len(not_results)))

        return

    demo_queries = [
        "aaron abandon",
        "president election",
        "football team",
        "market economy",
        "health care",
    ]

    rows = benchmark_queries(demo_queries, inverted_index, compressed_index)
    print_benchmark_table(rows)

    if args.mode == "benchmark":
        return

    print("\n================ Demo Search ================")
    sample_query = "president election"
    print("Sample query: " + sample_query)
    result = and_search_optimized(sample_query, inverted_index)
    print("Hits: {}".format(len(result)))
    for doc in result[:10]:
        print("  " + doc)
    if len(result) > 10:
        print("  ... ({} more)".format(len(result) - 10))


if __name__ == "__main__":
    main()
