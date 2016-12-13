#ifndef RNN_GRAMMAR_H
#define RNN_GRAMMAR_H
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include "cnn/dict.h"

using std::endl;
using std::cerr;

void load_dict(cnn::Dict &d, std::string fname);
void save_dict(cnn::Dict &d, std::string fname);
int ReadSentenceView(const std::string& line, cnn::Dict &dict, std::vector<int> &sent);
int ReadSentenceMulti(const std::string& line, cnn::Dict &dict,
        std::vector<int> &sent_val, std::vector<unsigned> &sent_offset, bool split);

struct Sentence;
struct Corpus;

struct RNNGrammar {
  // dictionaries
  cnn::Dict termdict, ntermdict, adict, posdict;
  cnn::Dict edgedict;
  // pass in index of action NT(X), return index of X
  std::map<unsigned,unsigned> action2NTindex;   
  std::vector<unsigned> possible_actions;
  Corpus *load_corpus(const std::string &fname, bool split);

  void Freeze() {
    termdict.Freeze();
    // prevents problems with the lowercased data
    termdict.SetUnk("UNK");
    adict.Freeze();
    ntermdict.Freeze();
    posdict.Freeze();
    edgedict.Freeze();
    cerr << "    cumulative      action vocab size: "
	 << adict.size() << endl;
    cerr << "    cumulative    terminal vocab size: "
	 << termdict.size() << endl;
    cerr << "    cumulative nonterminal vocab size: "
	 << ntermdict.size() << endl;
    cerr << "    cumulative         pos vocab size: "
	 << posdict.size() << endl;
    cerr << "    cumulative  func label vocab size: "
	 << edgedict.size() << endl;
    // compute action2NTindex
    for (unsigned i = 0; i < adict.size(); ++i) {
      const std::string& a = adict.Convert(i);
      if (a[0] != 'N') continue;
      size_t start = a.find('(') + 1;
      size_t end = a.rfind(')');
      int nt = ntermdict.Convert(a.substr(start, end - start));
      action2NTindex[i] = nt;
    }
    possible_actions.resize(adict.size());
    for (unsigned i = 0; i < adict.size(); ++i) {
      possible_actions[i] = i;
    }
  }

  void init_from_files(const std::string prefix) {
    load_dict(termdict, prefix+"_term.txt");
    load_dict(adict, prefix+"_actions.txt");
    load_dict(ntermdict, prefix+"_nterm.txt");
    load_dict(posdict, prefix+"_pos.txt");
    Freeze();
  }

  void save_to_files(const std::string prefix) {
    save_dict(termdict, prefix+"_term.txt");
    save_dict(adict, prefix+"_actions.txt");
    save_dict(ntermdict, prefix+"_nterm.txt");
    save_dict(posdict, prefix+"_pos.txt");
  }

  template<typename Stream>
    void write_tree(Stream &stream,
        const std::vector<unsigned> &actions,
        const Sentence &sentence);

  template<typename Stream>
    void write_tree(Stream &out,
        const std::vector<unsigned> &actions,
        const std::vector<unsigned> &edges,
        const Sentence &sentence);
};

/** a sentence can be viewed in 4 different ways:
    raw tokens, UNKed, lowercased, and POS tags
*/
struct Sentence {
  bool SizesMatch() const {
    return (raw.size() == unk.size() &&
	    raw.size() == lc.size() &&
	    raw.size() == (pos_offset.size()-1) &&
            pos_offset[0] == 0 &&
            pos_offset[raw.size()] == pos_val.size());
  }
  size_t size() const { return raw.size(); }
  std::vector<int> raw, unk, lc;
  std::vector<unsigned> pos_offset; std::vector<int> pos_val;
};

struct Corpus {
  RNNGrammar *grammar;
  Corpus(RNNGrammar *g)  {grammar=g;}
  unsigned size() const { return sents.size(); }
  std::vector<Sentence> sents;
  std::vector<std::vector<unsigned>> actions;
  std::vector<std::vector<unsigned>> edge_labels;
  void count(std::unordered_map<unsigned, unsigned> &counts) {
    for (auto& sent : sents)
      for (auto word : sent.raw) counts[word]++;
  }
  void load_oracle(const std::string& file, bool split);
};
  
template<typename Stream> void RNNGrammar::write_tree(Stream &out,
				     const std::vector<unsigned> &actions,
				     const Sentence &sentence) {
  int ti = 0;
  for (unsigned action: actions) {
    const std::string &s_action = adict.Convert(action);
    if (s_action[0] == 'N') {
      out << '(' << ntermdict.Convert(action2NTindex.find(action)->second)
	  << ' ';
    } else if (s_action[0] == 'S') {
      out << "(XX " << termdict.Convert(sentence.unk[ti++]) << ") ";
    } else { // REDUCE
      out << ") ";
    }
  }
  out << endl;
}

template<typename Stream> void RNNGrammar::write_tree(Stream &out,
				     const std::vector<unsigned> &actions,
                                     const std::vector<unsigned> &edges,
				     const Sentence &sentence) {
  int ti = 0;
  int edge_idx = 0;
  std::vector<std::string> output;
  std::vector<int> stack_content;
  std::vector<int> brackets;
  for (unsigned action: actions) {
    const std::string &s_action = adict.Convert(action);
    if (s_action[0] == 'N') {
      brackets.push_back(stack_content.size());
      stack_content.push_back(output.size());
      output.push_back("("+ntermdict.Convert(action2NTindex.find(action)->second));
    } else if (s_action[0] == 'S') {
      stack_content.push_back(output.size());
      output.push_back("(XX");
      output.push_back(termdict.Convert(sentence.unk[ti++])+")");
    } else { // REDUCE
      int stack_limit = brackets.back() + 1;
      brackets.pop_back();
      for (int i = stack_limit; i < stack_content.size(); i++) {
        output[stack_content[i]] += ":" + edgedict.Convert(edges[edge_idx++]);
      }
      stack_content.resize(stack_limit);
      output.push_back(")");
    }
  }
  for (int i=0; i<output.size(); i++) {
    if (i>0) out << ' ';
    out << output[i];
  }
  out << endl;
}
#endif /* RNN_GRAMMAR */
