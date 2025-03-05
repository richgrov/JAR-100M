import collections

class Tokenizer:
    def __init__(self, vocab: str, target_size: int):
        self.target_size = target_size
        self.tokens_dict = {}
        self.data = self.pre_tokenizer(vocab)

        while len(self.tokens_dict) < self.target_size:
            pair_counts = self.count_pairs(self.data)
            if not pair_counts:
                break
            
            best_pair = max(pair_counts, key=pair_counts.get)
            self.tokens_dict[best_pair] = len(self.tokens_dict)
            self.data = self.merge_pairs(self.data, best_pair)

        print(self.tokens_dict)
        # print(self.data)

    def pre_tokenizer(self, data: str):
        return [" ".join(" " + word) for word in data.split()]

    def count_pairs(self, words):
        pairs = collections.defaultdict(int)
        for word in words:
            chars = word.split()
            for i in range(len(chars) - 1):
                first = chars[i]
                # first = " " + chars[i] if i == 0 else first
                pair = (first, chars[i + 1])
                pairs[pair] += 1
        return pairs

    def merge_pairs(self, words, pair):
        new_words = []
        unmerged_token = " ".join(pair)
        merged_token = ''.join(pair)
        
        for word in words:
            word = word.replace(unmerged_token, merged_token)
            new_words.append(word)
        return new_words

def token(file_path: str):
    with open(file_path, "rb") as file:
        data = file.read()
    tokens = collections.defaultdict(int)
    text = data.split(b"\0")
    text = [t.decode('utf-8') for t in text]
    text.pop()
    i = 0
    while i < int(len(text)):
        tokens[(text[i], text[i + 1])] = int(i / 2)
        i += 2
    for token in tokens:
        print(token, tokens[token])


if __name__ == "__main__":
    # with open("dataset.txt", "r", encoding="utf-8") as file:
    #     data = file.read()
    #     print(len(data))
    # tokenizer = Tokenizer(data, 25)

    token("../Tokenizer/Tokenizer/tokens.bin")

