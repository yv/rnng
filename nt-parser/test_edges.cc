#include <iostream>
#include "rnn_grammar.h"

using namespace std;

int main(int argc, char **argv)
{
  RNNGrammar gram;
  Corpus *corp;
  if (argc > 1) {
    corp = gram.load_corpus(argv[1], false);
  } else {
    corp = gram.load_corpus("test.oracle", false);
  }
  gram.Freeze();
  for (int i=0; i<3; i++) {
    gram.write_tree(cout, corp->actions[i], corp->edge_labels[i],
        corp->sents[i]);
  }
}
