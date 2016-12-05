#ifndef PARSER_ORACLE_H_
#define PARSER_ORACLE_H_

#include <iostream>
#include <vector>
#include <string>

namespace cnn { class Dict; }

namespace parser {

// a sentence can be viewed in 4 different ways:
//   raw tokens, UNKed, lowercased, and POS tags
struct Sentence {
  bool SizesMatch() const { return raw.size() == unk.size() && raw.size() == lc.size() && raw.size() == pos.size(); }
  size_t size() const { return raw.size(); }
  std::vector<int> raw, unk, lc, pos;
};

// base class for transition based parse oracles
struct Oracle {
  RNNGrammar *gram;
Oracle(RNNGrammar *g) : gram(g), sents() {}
  virtual ~Oracle();
  unsigned size() const { return sents.size(); }
  std::string devdata;
  std::vector<Sentence> sents;
  std::vector<std::vector<int>> actions;
 protected:
  static void ReadSentenceView(const std::string& line, cnn::Dict* dict, std::vector<int>* sent);
};

// oracle that predicts nonterminal symbols with a NT(X) action
// the action NT(X) effectively introduces an "(X" on the stack
// # (S (NP ...
// raw tokens
// tokens with OOVs replaced
class TopDownOracle : public Oracle {
 public:
  TopDownOracle(RNNGrammar *gram) :
      Oracle(gram) {}
  // if is_training is true, then both the "raw" tokens and the mapped tokens
  // will be read, and both will be available. if false, then only the mapped
  // tokens will be available
  void load_bdata(const std::string& file);
  void load_oracle(const std::string& file, bool is_training);
};

// oracle that predicts nonterminal symbols with a NT(X) action
// the action NT(X) effectively introduces an "(X" on the stack
// # (S (NP ...
// raw tokens
// tokens with OOVs replaced
class TopDownOracleGen : public Oracle {
 public:
  TopDownOracleGen(RNNGrammar *gram) :
      Oracle(gram) {}
  void load_oracle(const std::string& file);
};

class TopDownOracleGen2 : public Oracle {
 public:
  TopDownOracleGen2(RNNGrammar *gram) :
      Oracle(gram) {}
  void load_oracle(const std::string& file);
};

} // namespace parser

#endif
