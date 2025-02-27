class Tokenizer:
    def __init__(self, vocab: str, target_size: int):
        self.tokens_set = set(vocab)
        self.tokens_set.add('he')
        self.data = vocab
        pass
    
    def pre_tokenizer(self, data: str) -> list:
        data = data.split(" ")
        for string in data:
            string += " "
        self.frequency(data, {})
        
    def frequency(self, words, dictionary: dict, iterations = 0):
        iterations += 1
        dictionary
        for string in words:
            sub_dict = {}
            for char_idx in range(len(string)):
                for char_slicer in range(char_idx, len(string)):
                    slice = string[char_idx:char_slicer + 2]
                    if slice not in sub_dict and slice not in dictionary:
                        sub_dict[slice] = 0
                    if slice not in dictionary:
                        sub_dict[slice] += 1
            highest: tuple[str, int] = ("", 0)
            for combo in sub_dict:
                if sub_dict[combo] > highest[1]:
                    highest = (combo, sub_dict[combo])
            dictionary[highest[0]] = highest[1]

        if len(dictionary) >= 100:
            print(dictionary)
            return
        self.frequency(words, dictionary, iterations)



    # def frequency(self):
    #     temp = []
    #     for i in range(len(self.data) - 1):
    #         temp.append((self.data[i], self.data[i + 1]))
    #     frequencies = []
    #     already_checked = []
    #     for i in range(len(temp)):
    #         count = 0
    #         for j in range(i, len(temp)):
    #             if temp[i] == temp[j] and temp[i] not in already_checked:
    #                 count += 1
    #         if temp[i] not in already_checked:
    #             already_checked.append(temp[i])
    #             frequencies.append((temp[i], count))
    #     print(len(temp), len(frequencies))
    #     print(frequencies)
    
    def merge(self, tokens: tuple[str, str]) -> str:
        pass
    
if __name__ == "__main__":
    tokenizer = Tokenizer("hello there I want to o o o o o o o o be a cowboy baby", 100)
    tokenizer.pre_tokenizer("hellohellohellohellohello jfhdsaoifuyasod there I want to o o o o o o o o be a cowboy baby")