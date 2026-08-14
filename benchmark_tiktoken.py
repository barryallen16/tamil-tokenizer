#!/usr/bin/env python3
import tiktoken
import time

def format_duration(nanos):
    if nanos < 1_000:
        return f"{nanos} ns"
    elif nanos < 1_000_000:
        return f"{nanos / 1_000:.2f} µs"
    elif nanos < 1_000_000_000:
        return f"{nanos / 1_000_000:.2f} ms"
    else:
        return f"{nanos / 1_000_000_000:.2f} s"

def format_throughput(bytes_size, nanos):
    mb = bytes_size / (1024 * 1024)
    secs = nanos / 1_000_000_000
    return f"{mb / secs:.2f} MB/s"

def load_corpus(filepath):
    """Load corpus from file (one document per line)"""
    texts = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                texts.append(line)
    return texts

def main():
    print("=== Tiktoken Corpus Benchmark (cl100k_base) ===\n")
    
    start = time.perf_counter_ns()
    enc = tiktoken.get_encoding("cl100k_base")
    build_time = time.perf_counter_ns() - start
    print(f"Encoder build time: {format_duration(build_time)}\n")
    
    # Load corpus
    corpus_path = "./data/tamil_corpus.txt"
    try:
        corpus_texts = load_corpus(corpus_path)
        print(f"Loaded {len(corpus_texts)} documents from corpus\n")
    except FileNotFoundError:
        print(f"Corpus file not found: {corpus_path}")
        return
    
    # Benchmark corpus
    print("┌──────────┬─────────┬──────────────┬────────────────────────┐")
    print("│ Doc ID   │ Chars   │ Tokens       │ Time / Throughput      │")
    print("├──────────┼─────────┼──────────────┼────────────────────────┤")
    
    total_bytes = 0
    total_tokens = 0
    total_time_nanos = 0
    
    sample_size = min(len(corpus_texts), 100)
    
    for idx, text in enumerate(corpus_texts[:sample_size]):
        char_count = len(text)
        byte_count = len(text.encode('utf-8'))
        
        # Warmup
        for _ in range(3):
            _ = enc.encode(text)
        
        # Benchmark
        start = time.perf_counter_ns()
        tokens = enc.encode(text)
        encode_time = time.perf_counter_ns() - start
        
        total_bytes += byte_count
        total_tokens += len(tokens)
        total_time_nanos += encode_time
        
        throughput = format_throughput(byte_count, encode_time)
        
        print(f"│ {idx+1:>8} │ {char_count:>7} │ {len(tokens):>12} │ {format_duration(encode_time):>8} {throughput:>11} │")
        
        if (idx + 1) % 10 == 0:
            print("├──────────┼─────────┼──────────────┼────────────────────────┤")
    
    print("└──────────┴─────────┴──────────────┴────────────────────────┘")
    
    # Summary
    print("\n=== Corpus Summary ===")
    print(f"Documents processed: {sample_size}")
    print(f"Total bytes: {total_bytes}")
    print(f"Total tokens: {total_tokens}")
    print(f"Average compression ratio: {total_bytes / total_tokens:.2f} bytes/token")
    print(f"Overall throughput: {format_throughput(total_bytes, total_time_nanos)}")
    print(f"Average time per document: {format_duration(total_time_nanos // sample_size)}")
    
    # Full batch
    print("\n=== Batch Processing (all corpus) ===")
    start = time.perf_counter_ns()
    for text in corpus_texts:
        _ = enc.encode(text)
    batch_time = time.perf_counter_ns() - start
    
    print(f"Batch size: {len(corpus_texts)} documents")
    print(f"Total time: {format_duration(batch_time)}")
    print(f"Per document: {format_duration(batch_time // len(corpus_texts))}")

if __name__ == "__main__":
    main()
