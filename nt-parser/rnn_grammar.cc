#include <string>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include "nt-parser/compressed-fstream.h"
#include "nt-parser/rnn_grammar.h"

using std::string;
using std::ofstream;

const string kREDUCE = "REDUCE";
const string kSHIFT = "SHIFT";

void load_dict(cnn::Dict &d, const string &fname) {
  assert(&d!=NULL);
  assert(d.size() == 0);
  cnn::compressed_ifstream in(fname.c_str());
  string line;
  getline(in, line);
  int n = atoi(line.c_str());
  for (int i=0; i<n; i++) {
    getline(in, line);
    d.Convert(line);
  }
  d.Freeze();
}

void save_dict(cnn::Dict &d, const string &fname) {
  assert(&d!=NULL);
  ofstream out(fname.c_str());
  out << d.size() << endl;
  for (int i=0; i<d.size(); i++) {
    out << d.Convert(i) << endl;
  }
}

inline bool is_ws(char x) { //check whether the character is a space or tab delimiter
  return (x == ' ' || x == '\t');
}

inline bool is_ws_or_pipe(char x) {
  return (x == ' ' || x == '\t' || x == '|');
}

inline bool is_not_ws(char x) {
  return (x != ' ' && x != '\t');
}

int ReadSentenceView(const string& line, cnn::Dict &dict, std::vector<int> &sent) {
  unsigned cur = 0;
  while(cur < line.size()) {
    while(cur < line.size() && is_ws(line[cur])) { ++cur; }
    unsigned start = cur;
    while(cur < line.size() && is_not_ws(line[cur])) { ++cur; }
    unsigned end = cur;
    if (end > start) {
      unsigned x = dict.Convert(line.substr(start, end - start));
      sent.push_back(x);
    }
  }
  return sent.size();
}

int ReadSentenceMulti(const std::string& line, cnn::Dict &dict,
    std::vector<int> &sent_val, std::vector<unsigned> &sent_offset, bool split) {
  unsigned cur = 0;
  sent_offset.push_back(0);
  while(cur < line.size()) {
    while(cur < line.size() && is_ws_or_pipe(line[cur])) {
      ++cur;
    }
    unsigned start = cur; 
    while(cur < line.size() &&
        is_not_ws(line[cur]) && (!split || line[cur]!='|')) {
      ++cur;
    }
    unsigned end = cur;
    if (end > start) {
      unsigned x = dict.Convert(line.substr(start, end - start));
      sent_val.push_back(x);
    }
    if (cur == line.size() || is_ws(line[cur])) {
      sent_offset.push_back(sent_val.size()); 
    }
  }
  assert(sent_val.size() > 0); // empty sentences not allowed
  return sent_offset.size() - 1;
}

Corpus *RNNGrammar::load_corpus(const string& fname, bool split) {
  Corpus *corpus = new Corpus(this);
  corpus->load_oracle(fname, split);
  return corpus;
}


void Corpus::load_oracle(const string& file, bool split) {
  cerr << "Loading top-down oracle from " << file << " ...\n";
  cnn::compressed_ifstream in(file.c_str());
  assert(in);
  const int kREDUCE_INT = grammar->adict.Convert(kREDUCE);
  const int kSHIFT_INT = grammar->adict.Convert(kSHIFT);
  int lc = 0;
  string line;
  std::vector<unsigned> cur_acts;
  std::vector<unsigned> cur_edges;
  while(getline(in, line)) {
    ++lc;
    //cerr << "line number = " << lc << endl;
    cur_acts.clear();
    cur_edges.clear();
    if (line.size() == 0 || line[0] == '#') continue;
    sents.resize(sents.size() + 1);
    auto& cur_sent = sents.back();
    ReadSentenceMulti(line, grammar->posdict,
        cur_sent.pos_val, cur_sent.pos_offset, split);
    getline(in, line);
    ReadSentenceView(line, grammar->termdict, cur_sent.raw);
    getline(in, line);
    ReadSentenceView(line, grammar->get_pretrain_dict(), cur_sent.lc);
    getline(in, line);
    ReadSentenceView(line, grammar->termdict, cur_sent.unk);
    lc += 3;
    if (!cur_sent.SizesMatch()) {
      cerr << "Mismatched lengths of input strings in oracle before line " << lc << endl;
      abort();
    }
    if (cur_sent.size() == 0) {
      cerr << "Empty sentence near line " << lc << endl;
      abort();
    }
    int termc = 0;
    int n_const = 0;
    while(getline(in, line)) {
      ++lc;
      //cerr << "line number = " << lc << endl;
      if (line.size() == 0) break;
      if (line[0] == 'R') {
        std::vector<string> symbols;
        boost::split(symbols, line, boost::is_any_of("\t "));
        if (symbols[0] != kREDUCE) {
          cerr << "Malformed input in line " << lc << endl;
          abort();
        }
        for (int j=1; j<symbols.size(); j++) {
          cur_edges.push_back(grammar->edgedict.Convert(symbols[j]));
        }
        cur_acts.push_back(kREDUCE_INT);
      } else if (line.find("NT(") == 0) {
        assert(line.find(' ') == string::npos);
        n_const++;
        // Convert NT
        grammar->ntermdict.Convert(line.substr(3, line.size() - 4));
        // NT(X) is put into the actions list as NT(X)
        cur_acts.push_back(grammar->adict.Convert(line));
      } else if (line == kSHIFT) {
        cur_acts.push_back(kSHIFT_INT);
        termc++;
        n_const++;
      } else {
        cerr << "Malformed input in line " << lc << endl;
        abort();
      }
    }
    actions.push_back(cur_acts);
    edge_labels.push_back(cur_edges);
    if (termc != sents.back().size()) {
      cerr << "Mismatched number of tokens and SHIFTs in oracle before line " << lc << endl;
      abort();
    }
    if (cur_edges.size() != 0 && cur_edges.size() != n_const - 1) {
      cerr << "Weird number of edge labels in oracle before line " << lc << endl;
    }
  }
  cerr << "Loaded " << sents.size() << " sentences\n";
}
