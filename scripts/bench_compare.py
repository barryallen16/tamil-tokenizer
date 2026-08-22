#!/usr/bin/env python3
"""Benchmark tiktoken on Tamil Wikipedia text for comparison."""
import time
import tiktoken

def format_duration(nanos):
    if nanos < 1_000:
        return f"{nanos} ns"
    elif nanos < 1_000_000:
        return f"{nanos / 1_000:.2f} µs"
    elif nanos < 1_000_000_000:
        return f"{nanos / 1_000_000:.2f} ms"
    else:
        return f"{nanos / 1_000_000_000:.2f} s"

def main():
    print("=== Tiktoken on Tamil Wikipedia Text ===\n")
    
    # Load encoder
    start = time.perf_counter_ns()
    enc = tiktoken.get_encoding("cl100k_base")
    build_time = time.perf_counter_ns() - start
    print(f"Encoder build time: {format_duration(build_time)}\n")
    
    # Load Tamil corpus
    with open("data/tamil_wiki_corpus.txt", "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    
    # Sample 100 lines
    sample = lines[:100]
    print(f"Loaded {len(lines)} lines, benchmarking first {len(sample)}\n")
    
    total_bytes = 0
    total_tokens = 0
    total_time = 0
    
    print("┌──────────┬─────────┬──────────────┬────────────────────────┐")
    print("│ Line ID  │ Chars   │ Tokens       │ Time / Throughput      │")
    print("├──────────┼─────────┼──────────────┼────────────────────────┤")
    
    for i, text in enumerate(sample):
        byte_count = len(text.encode('utf-8'))
        
        # Warmup
        for _ in range(3):
            _ = enc.encode(text)
        
        start = time.perf_counter_ns()
        tokens = enc.encode(text)
        elapsed = time.perf_counter_ns() - start
        
        total_bytes += byte_count
        total_tokens += len(tokens)
        total_time += elapsed
        
        if (i + 1) % 10 == 0:
            mb = byte_count / (1024 * 1024)
            secs = elapsed / 1e9
            throughput = mb / secs if secs > 0 else 0
            print(f"│ {i+1:>8} │ {len(text):>7} │ {len(tokens):>12} │ {format_duration(elapsed):>8} {throughput:>8.2f} MB/s │")
    
    print("└──────────┴─────────┴──────────────┴────────────────────────┘")
    
    # Summary
    mb = total_bytes / (1024 * 1024)
    secs = total_time / 1e9
    throughput = mb / secs if secs > 0 else 0
    
    print(f"\n=== Summary ===")
    print(f"Lines processed: {len(sample)}")
    print(f"Total bytes: {total_bytes:,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Compression ratio: {total_bytes/total_tokens:.2f} bytes/token")
    print(f"Overall throughput: {throughput:.2f} MB/s")
    print(f"Avg time per line: {format_duration(total_time // len(sample))}")

if __name__ == '__main__':
    main()
