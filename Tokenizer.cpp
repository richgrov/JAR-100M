#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

using Token = uint16_t;
using TokenPair = uint32_t;

namespace timing = std::chrono;
using tuples_map = std::unordered_map<TokenPair, Token>;
using word = std::vector<int>;

class Tokenizer {
public:
  Tokenizer(int target_size, const std::string &data);
  ~Tokenizer();

private:
  int _target_size;
  std::vector<Token> _data_tokenized;
  std::vector<TokenPair> _tokens;
  std::map<std::string, Token> _vocab;
  std::vector<std::string> _inv_vocab;

  std::vector<Token> pre_tokenizer(const std::string &data);
  tuples_map count_pairs(const std::vector<Token> &data);
  TokenPair find_max(const tuples_map &tuples);
  void merge_pair(TokenPair pair, std::vector<Token> &words);
  Token get_id(const std::string &token);
};

Tokenizer::Tokenizer(int target_size, const std::string &data)
    : _target_size(target_size) {
  auto begin = timing::steady_clock::now();
  _data_tokenized = pre_tokenizer(data);

  for (int i = 0; i < target_size; i++) {
    auto start = timing::steady_clock::now();

    tuples_map to_max = count_pairs(_data_tokenized);
    TokenPair max = find_max(to_max);
    _tokens.push_back(max);
    merge_pair(max, _data_tokenized);

    auto end = timing::steady_clock::now();
    std::cout
        << timing::duration_cast<timing::milliseconds>(end - start).count()
        << ",  ";
  }

  std::ofstream token_file;
  token_file.open("tokens.bin", std::ios::binary);
  for (const auto &token : _tokens) {
    token_file << _inv_vocab[token >> 16]
               << std::string(1, static_cast<char>(0))
               << _inv_vocab[token & 0xFFFF]
               << std::string(1, static_cast<char>(0));
  }
  token_file.close();
  auto finish = timing::steady_clock::now();
  auto tim =
      timing::duration_cast<timing::milliseconds>(finish - begin).count();
  auto to_seconds = (tim / 1000) % 60;
  std::cout << "Total time: " << tim / 60000 << " minutes, " << to_seconds
            << " seconds, " << tim % 1000 << " milliseconds."
            << "\n";
}

Tokenizer::~Tokenizer() {}

std::vector<Token> Tokenizer::pre_tokenizer(const std::string &data) {
  std::string _word;
  std::vector<Token> tokens_fully_split;

  word chars_split;
  for (char c : data) {
    tokens_fully_split.emplace_back(get_id(std::string(1, c)));
  }
  return tokens_fully_split;
}

tuples_map Tokenizer::count_pairs(const std::vector<Token> &data) {
  tuples_map pairs;
  for (size_t i = 0; i < data.size() - 1; i++) {
    pairs[data[i] << 16 | data[i + 1]]++;
  }
  return pairs;
}

TokenPair Tokenizer::find_max(const tuples_map &tuples) {
  return std::max_element(
             tuples.begin(), tuples.end(),
             [](const auto &a, const auto &b) { return a.second < b.second; })
      ->first;
}

void Tokenizer::merge_pair(TokenPair pair, std::vector<Token> &words) {
  uint16_t left = pair >> 16;
  uint16_t right = pair & 0xFFFF;
  uint32_t merged_id = get_id(_inv_vocab[left] + _inv_vocab[right]);

  size_t write_index = 0;
  for (size_t i = 0; i < words.size() - 1; i++, write_index++) {
    if ((words[i] << 16 | words[i + 1]) == pair) {
      words[write_index] = merged_id;
      i++;
    } else {
      words[write_index] = words[i];
    }
  }

  words.resize(write_index + 1);
}

Token Tokenizer::get_id(const std::string &token) {
  auto it = _vocab.find(token);
  if (it == _vocab.end()) {
    Token id = _vocab.size();
    _vocab[token] = id;
    _inv_vocab.emplace_back(token);
    return id;
  }
  return it->second;
}

int main() {
  std::ifstream file("/Users/richard/Documents/lllm/dataset.txt");
  std::string file_contents((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
  Tokenizer tokenizer(1000, file_contents);
}
