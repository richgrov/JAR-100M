#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <unordered_map>
#include <functional>

struct tuple_hash {
	std::size_t operator()(const std::pair<int, int>& p) const {
		return std::hash<int>{}(p.first) ^ (std::hash<int>{}(p.second) << 1);
	}
};

#define timing std::chrono
using tuple = std::pair<int, int>;
using tuples_map = std::unordered_map<tuple, int, tuple_hash>;
using word = std::vector<int>;


class Tokenizer {
public:
	Tokenizer(int target_size, const std::string& data);
	~Tokenizer();

private:
	int _target_size;
	std::vector<int> _data_tokenized;
	std::vector<tuple> _tokens;
	std::map<std::string, int> _vocab;
	std::map<int, std::string> _inv_vocab;
	int _next_id;

	std::vector<int> pre_tokenizer(const std::string& data);
	word split_word(const std::string& _word);
	tuples_map count_pairs(const std::vector<int>& data);
	tuple find_max(const tuples_map& tuples);
	void merge_pair(const tuple& pair, std::vector<int>& words);
	int get_id(const std::string& token);
};

Tokenizer::Tokenizer(int target_size, const std::string& data) : _target_size(target_size), _next_id(0) {
	auto begin = timing::steady_clock::now();
	_data_tokenized = pre_tokenizer(data);

	for (int i = 0; i < 10000; i++) {
		auto start = timing::steady_clock::now();

		tuples_map to_max = count_pairs(_data_tokenized);
		tuple max = find_max(to_max);
		_tokens.push_back(max);
		merge_pair(max, _data_tokenized);

		auto end = timing::steady_clock::now();
		std::cout << timing::duration_cast<timing::milliseconds>(end - start).count() << ",  ";
	}

	std::ofstream token_file;
	token_file.open("tokens.bin", std::ios::binary);
	for (const auto& token : _tokens) {
		token_file << _inv_vocab[token.first] << std::string(1, static_cast<char>(0)) << _inv_vocab[token.second] << std::string(1, static_cast<char>(0));
	}
	token_file.close();
	auto finish = timing::steady_clock::now();
	auto tim = timing::duration_cast<timing::milliseconds>(finish - begin).count();
	auto to_seconds = (tim / 1000) % 60;
	std::cout << "Total time: " << tim / 60000 << " minutes, " << to_seconds << " seconds, " << tim % 1000 << " milliseconds." << "\n";
}

Tokenizer::~Tokenizer() {}

std::vector<int> Tokenizer::pre_tokenizer(const std::string& data) {
	std::string _word;
	std::vector < int > tokens_fully_split;

	word chars_split;
	for (char c : data) {
		tokens_fully_split.emplace_back(get_id(std::string(1, c)));
	}
	return tokens_fully_split;
}

word Tokenizer::split_word(const std::string& _word) {}

tuples_map Tokenizer::count_pairs(const std::vector<int>& data) {
	tuples_map pairs;
	for (size_t i = 0; i < data.size(); i++) {
		pairs[{data[i], data[i + 1]}]++;
	}
	return pairs;
}

tuple Tokenizer::find_max(const tuples_map& tuples) {
	return std::max_element(tuples.begin(), tuples.end(),
		[](const auto& a, const auto& b) {
			return a.second < b.second;
		})->first;
}

void Tokenizer::merge_pair(const tuple& pair, std::vector<int>& words) {
	int pair_first = pair.first;
	int pair_second = pair.second;
	int merged_id = get_id(_inv_vocab[pair_first] + _inv_vocab[pair_second]);

	for (size_t i = 0; i < words.size() - 1; i++) {

		if (words[i] == pair_first && words[i + 1] == pair_second) {
			words[i] = merged_id;
			words.erase(words.begin() + (i + 1));
			i++;
		}
	}
}

int Tokenizer::get_id(const std::string& token) {
	auto it = _vocab.find(token);
	if (it == _vocab.end()) {
		int id = _next_id++;
		_vocab[token] = id;
		_inv_vocab[id] = token;
		return id;
	}
	return it->second;
}

int main() {
	std::ifstream file("../../lllm/dataset.txt");
	std::string file_contents((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
	Tokenizer tokenizer(128, file_contents);
}
