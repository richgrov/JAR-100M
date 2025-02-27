class Tokenizer:
    def __init__(self, vocab: str, target_size: int):
        self.target_size = target_size
        self.tokens_set = set(vocab)
        self.tokens_set.add('he')
        self.data = vocab
        self.tokens_dict = {}
        while len(self.tokens_dict) < self.target_size:
            self.tokens_dict = self.frequency(self.pre_tokenizer(vocab), self.tokens_dict)
        # print(self.tokens_dict)
    
    def pre_tokenizer(self, data: str) -> list:
        data = data.split(" ")
        for string in data:
            string += " "
        return data
    
    def frequency(self, words, dictionary: dict):
        dictionary
        for string in words:
            sub_dict = {}
            for char_idx in range(len(string)):
                for char_slicer in range(char_idx, len(string) - 1):
                    slice = string[char_idx:char_slicer + 2]
                    if slice not in sub_dict and slice not in dictionary:
                        sub_dict[slice] = 0
                    if slice not in dictionary:
                        sub_dict[slice] += 1
                    if slice in dictionary:
                        dictionary[slice] += 1
            highest: tuple[str, int] = ("", -11)
            for combo in sub_dict:
                if sub_dict[combo] > highest[1]:
                    highest = (combo, sub_dict[combo])
            if highest[0] not in dictionary and highest[0] != "":
                dictionary[highest[0]] = highest[1]
            if highest[0] != "":
                dictionary[highest[0]] += highest[1]
        return dictionary

if __name__ == "__main__":
    file = open("dataset.txt")
    data = file.read()
    tokenizer = Tokenizer(data, 5000)
    # tokenizer = Tokenizer("th\ne the the the thing that I made the thing is there", 100)