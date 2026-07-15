import json

def build():
    text = open("../data/train_corpus.txt", encoding="utf-8").read()
    chars = set(text)
    
    ch_to_id = {}
    next_id = 256
    
    for ch in sorted(chars):
        if len(ch.encode("utf-8")) > 1:
            ch_to_id[ch] = next_id
            next_id += 1
            
    with open("vocab.json", "w", encoding="utf-8") as f:
        json.dump(ch_to_id, f, ensure_ascii=False)
        
    print(f"Built vocab with {len(ch_to_id)} multi-byte characters.")
    print(f"Total vocab size will be {256 + len(ch_to_id)}")

if __name__ == "__main__":
    build()
