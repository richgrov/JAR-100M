#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <map>

#define timing std::chrono
using tuple = std::pair<int, int>;
using tuples_map = std::map<tuple, int>;
using word = std::vector<int>;

class Tokenizer {
public:
	Tokenizer(int target_size, const std::string& data);
	~Tokenizer();

private:
	int _target_size;
	std::vector<word> _data_tokenized;
	std::vector<tuple> _tokens;
	std::map<std::string, int> _vocab;
	std::map<int, std::string> _inv_vocab;
	int _next_id;

	std::vector<word> pre_tokenizer(const std::string& data);
	word split_word(const std::string& _word);
	tuples_map count_pairs(const std::vector<word>& data);
	tuple find_max(const tuples_map& tuples);
	void merge_pair(const tuple& pair, std::vector<word>& words);
	int get_id(const std::string& token);
};

Tokenizer::Tokenizer(int target_size, const std::string& data) : _target_size(target_size), _next_id(0) {
	auto begin = timing::steady_clock::now();
	_data_tokenized = pre_tokenizer(data);

	for (int i = 0; i < 5000; i++) {
		auto start = timing::steady_clock::now();

		tuples_map to_max = count_pairs(_data_tokenized);
		tuple max = find_max(to_max);
		_tokens.push_back(max);
		merge_pair(max, _data_tokenized);

		auto end = timing::steady_clock::now();
		std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;
	}

	for (const auto& token : _data_tokenized) {
		for (int c : token) {
			std::cout << _inv_vocab[c] << " ";
		}
		std::cout << std::endl;
	}
	auto finish = timing::steady_clock::now();
	std::cout << "Total time: " << std::chrono::duration_cast<std::chrono::milliseconds>(finish - begin).count() << " millis." << std::endl;
}

Tokenizer::~Tokenizer() {}

std::vector<word> Tokenizer::pre_tokenizer(const std::string& data) {
	std::istringstream iss(data);
	std::string _word;
	std::vector<word> tokens_fully_split;

	while (iss >> _word) {
		tokens_fully_split.emplace_back(split_word(_word));
	}
	return tokens_fully_split;
}

word Tokenizer::split_word(const std::string& _word) {
	word chars_split;
	for (char c : _word) {
		chars_split.emplace_back(get_id(std::string(1, c)));
	}
	return chars_split;
}

tuples_map Tokenizer::count_pairs(const std::vector<word>& data) {
	tuples_map pairs;
	for (const auto& _word : data) {
		size_t size = _word.size();
		if (size < 2) continue;

		for (size_t i = 0; i < size - 1; i++) {
			pairs[{_word[i], _word[i + 1]}]++;
		}
	}
	return pairs;
}

tuple Tokenizer::find_max(const tuples_map& tuples) {
	return std::max_element(tuples.begin(), tuples.end(),
		[](const auto& a, const auto& b) {
			return a.second < b.second;
		})->first;
}

void Tokenizer::merge_pair(const tuple& pair, std::vector<word>& words) {
	int pair_first = pair.first;
	int pair_second = pair.second;
	int merged_id = get_id(_inv_vocab[pair_first] + _inv_vocab[pair_second]);

	for (auto& _word : words) {
		size_t size = _word.size();
		if (size < 2) continue;

		word new_word;
		new_word.reserve(size);

		for (size_t i = 0; i < size; i++) {
			if (i < size - 1 && _word[i] == pair_first && _word[i + 1] == pair_second) {
				new_word.emplace_back(merged_id);
				i++;
			}
			else {
				new_word.emplace_back(_word[i]);
			}
		}
		_word = std::move(new_word);
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
