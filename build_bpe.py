import json

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

def build():
    text = open("../data/train_corpus.txt", encoding="utf-8").read()
    # Train on first 500,000 characters to save time
    train_text = text[:500000]
    ids = list(train_text.encode("utf-8"))
    
    num_merges = 768 # target vocab = 1024
    merges = {}
    
    print(f"Training BPE on {len(ids)} bytes...")
    for i in range(num_merges):
        stats = get_stats(ids)
        if not stats:
            break
        pair = max(stats, key=stats.get)
        idx = 256 + i
        ids = merge(ids, pair, idx)
        merges[f"{pair[0]},{pair[1]}"] = idx
        if (i+1) % 50 == 0:
            print(f"merge {i+1}/{num_merges}")
            
    with open("bpe_merges.json", "w", encoding="utf-8") as f:
        json.dump(merges, f)
    print("Done! Saved bpe_merges.json")

if __name__ == "__main__":
    build()
