import json
import os

class CharTokenizer:
    def __init__(self, ch_to_id=None):
        if ch_to_id is None:
            self.ch_to_id = {}
        else:
            self.ch_to_id = ch_to_id
        
        self.id_to_ch = {v: k for k, v in self.ch_to_id.items()}
        self.vocab_size = 256 + len(self.ch_to_id)
        
    def encode(self, text):
        ids = []
        for ch in text:
            if ch in self.ch_to_id:
                ids.append(self.ch_to_id[ch])
            else:
                ids.extend(list(ch.encode("utf-8")))
        return ids

    def decode(self, ids):
        parts = []
        i = 0
        while i < len(ids):
            if ids[i] >= 256:
                parts.append(self.id_to_ch[ids[i]].encode("utf-8"))
                i += 1
            else:
                b_list = []
                while i < len(ids) and ids[i] < 256:
                    b_list.append(ids[i])
                    i += 1
                parts.append(bytes(b_list))
        return b"".join(parts).decode("utf-8", errors="replace")

def load(path=None):
    if path is None:
        dir_path = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(dir_path, "vocab.json")
    
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            ch_to_id = json.load(f)
        return CharTokenizer(ch_to_id)
    else:
        return CharTokenizer()
