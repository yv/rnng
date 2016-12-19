#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include "nt-parser/rnn_grammar.h"
#include "nt-parser/parser_disc.h"

using std::ostringstream;
using std::ifstream;
using std::ofstream;
using cnn::rand01;

string NetworkSettings::propose_filename() {
  ostringstream os;
  os << "ntparse"
     << (USE_POS ? "_pos" : "")
     << '_' << IMPLICIT_REDUCE_AFTER_SHIFT
     << '_' << LAYERS
     << '_' << INPUT_DIM
     << '_' << HIDDEN_DIM
     << '_' << ACTION_DIM
     << '_' << LSTM_INPUT_DIM
     << "-pid" << getpid() << ".params";
  return os.str();
}

ParserBuilder::ParserBuilder(
        const RNNGrammar *gram,
        const NetworkSettings *settings,
        const vector<bool> &singleton_init,
        Model* model,
        unordered_map<unsigned, vector<float>>* pre) :
    grammar(gram),
    nt_size(gram->ntermdict.size()),
    action_size(gram->adict.size()),
    vocab_size(gram->termdict.size()),
    pretrain_size(gram->get_pretrain_dict().size()),
    pos_size(gram->posdict.size()),
    edges_size(gram->edgedict.size()),
    implicit_reduce(settings->IMPLICIT_REDUCE_AFTER_SHIFT),
    use_pos(settings->USE_POS),
    use_edges(settings->USE_EDGES),
    dropout_amount(settings->DROPOUT),
    alpha(settings->ALPHA),
    pretrained(pre), 
    singletons(singleton_init),
    // summaries of the stack and of previous actions
    stack_lstm(settings->LAYERS, settings->LSTM_INPUT_DIM,
            settings->HIDDEN_DIM, model),
    action_lstm(settings->LAYERS, settings->ACTION_DIM,
            settings->HIDDEN_DIM, model),
    // used to compose children (fwd/rev) into a representation of the node
    const_lstm_fwd(settings->LAYERS, settings->LSTM_INPUT_DIM,
            settings->LSTM_INPUT_DIM, model),
    const_lstm_rev(settings->LAYERS, settings->LSTM_INPUT_DIM,
            settings->LSTM_INPUT_DIM, model),
    p_w(model->add_lookup_parameters(vocab_size, {settings->INPUT_DIM})),
    p_t(model->add_lookup_parameters(pretrain_size, {settings->INPUT_DIM})),
    p_nt(model->add_lookup_parameters(nt_size, {settings->LSTM_INPUT_DIM})),
    p_ntup(model->add_lookup_parameters(nt_size, {settings->LSTM_INPUT_DIM})),
    p_a(model->add_lookup_parameters(action_size, {settings->ACTION_DIM})),
    p_pbias(model->add_parameters({settings->HIDDEN_DIM})),
    p_A(model->add_parameters({settings->HIDDEN_DIM, settings->HIDDEN_DIM})),
    p_B(model->add_parameters({settings->HIDDEN_DIM, settings->HIDDEN_DIM})),
    p_S(model->add_parameters({settings->HIDDEN_DIM, settings->HIDDEN_DIM})),
    p_w2l(model->add_parameters({settings->LSTM_INPUT_DIM, settings->INPUT_DIM})),
    p_ib(model->add_parameters({settings->LSTM_INPUT_DIM})),
    p_cbias(model->add_parameters({settings->LSTM_INPUT_DIM})),
    p_p2a(model->add_parameters({action_size, settings->HIDDEN_DIM})),
    p_action_start(model->add_parameters({settings->ACTION_DIM})),
    p_abias(model->add_parameters({action_size})),

    p_buffer_guard(model->add_parameters({settings->LSTM_INPUT_DIM})),
    p_stack_guard(model->add_parameters({settings->LSTM_INPUT_DIM})),

    p_cW(model->add_parameters({settings->LSTM_INPUT_DIM,
                settings->LSTM_INPUT_DIM * 2}))
{
    if (implicit_reduce) {
        // preterminal bias (used with IMPLICIT_REDUCE_AFTER_SHIFT)
        p_ptbias = model->add_parameters({settings->LSTM_INPUT_DIM});
        // preterminal W (used with IMPLICIT_REDUCE_AFTER_SHIFT)
        p_ptW = model->add_parameters({settings->LSTM_INPUT_DIM,
                2*settings->LSTM_INPUT_DIM}); 
    }
    if (use_pos) {
      p_pos = model->add_lookup_parameters(pos_size, {settings->POS_DIM});
      p_p2w = model->add_parameters({settings->LSTM_INPUT_DIM,
          settings->POS_DIM});
    }
    if (use_edges) {
      // for edge encoder
      p_Be = model->add_parameters({settings->HIDDEN_DIM, settings->HIDDEN_DIM});
      p_Se = model->add_parameters({settings->HIDDEN_DIM, settings->HIDDEN_DIM});
      p_eW = model->add_parameters({settings->HIDDEN_DIM,
          settings->LSTM_INPUT_DIM * 2});
      p_ebias = model->add_parameters({settings->HIDDEN_DIM});
      p_e2lbl = model->add_parameters({edges_size,
          settings->HIDDEN_DIM});
      p_lbias = model->add_parameters({edges_size});
    }
    buffer_lstm = new LSTMBuilder(settings->LAYERS,
            settings->LSTM_INPUT_DIM, settings->HIDDEN_DIM, model);
    if (pretrained->size() > 0) {
        p_t = model->add_lookup_parameters(vocab_size,
                {settings->PRETRAINED_DIM});
        for (auto it : *pretrained)
            p_t->Initialize(it.first, it.second);
        p_t2l = model->add_parameters({settings->LSTM_INPUT_DIM,
                settings->PRETRAINED_DIM});
    } else {
        p_t = nullptr;
        p_t2l = nullptr;
    }
}

bool IsActionForbidden_Discriminative(const string& a, char prev_a,
        unsigned bsize, unsigned ssize, unsigned nopen_parens,
        bool implicit_reduce)
{
    bool is_shift = (a[0] == 'S' && a[1]=='H');
    bool is_reduce = (a[0] == 'R' && a[1]=='E');
    bool is_nt = (a[0] == 'N');
    assert(is_shift || is_reduce || is_nt);
    static const unsigned MAX_OPEN_NTS = 100;
    if (is_nt && nopen_parens > MAX_OPEN_NTS) return true;
    if (ssize == 1) {
        if (!is_nt) return true;
        return false;
    }

    if (implicit_reduce) {
        // if a SHIFT has an implicit REDUCE, then only shift after an NT:
        if (is_shift && prev_a != 'N') return true;
    }

    // be careful with top-level parens- you can only close them if you
    // have fully processed the buffer
    if (nopen_parens == 1 && bsize > 1) {
        if (implicit_reduce && is_shift) return true;
        if (is_reduce) return true;
    }

    // you can't reduce after an NT action
    if (is_reduce && prev_a == 'N') return true;
    if (is_nt && bsize == 1) return true;
    if (is_shift && bsize == 1) return true;
    if (is_reduce && ssize < 3) return true;

    // TODO should we control the depth of the parse in some way? i.e., as long as there
    // are items in the buffer, we can do an NT operation, which could cause trouble
    return false;
}

// *** if correct_actions is empty, this runs greedy decoding ***
// returns parse actions for input sentence (in training just returns the reference)
// this lets us use pretrained embeddings, when available, for words that were OOV in the
// parser training data
// set sample=true to sample rather than max
vector<unsigned> ParserBuilder::log_prob_parser(ComputationGraph* hg,
        const Sentence& sent,
        const vector<unsigned>& correct_actions,
        const vector<unsigned>& correct_labels,
        vector<unsigned> *result_labels,
        double *right,
        bool is_evaluation,
        bool sample) {
  vector<unsigned> results;
  const bool build_training_graph = correct_actions.size() > 0;
  // Use cases:
  // - train both actions and edges [train_edges && !is_evaluation]
  // - train actions without edges (training example without edge labels)
  // - predict both actions and edges
  // - predict edges based on given actions
  const bool predict_edges = (is_evaluation && use_edges &&
      result_labels != NULL);
  const bool train_edges = (use_edges && correct_labels.size() > 0);
  bool apply_dropout = (dropout_amount && !is_evaluation);
  stack_lstm.new_graph(*hg);
  action_lstm.new_graph(*hg);
  const_lstm_fwd.new_graph(*hg);
  const_lstm_rev.new_graph(*hg);
  stack_lstm.start_new_sequence();
  buffer_lstm->new_graph(*hg);
  buffer_lstm->start_new_sequence();
  action_lstm.start_new_sequence();
  if (apply_dropout) {
    stack_lstm.set_dropout(dropout_amount);
    action_lstm.set_dropout(dropout_amount);
    buffer_lstm->set_dropout(dropout_amount);
    const_lstm_fwd.set_dropout(dropout_amount);
    const_lstm_rev.set_dropout(dropout_amount);
  } else {
    stack_lstm.disable_dropout();
    action_lstm.disable_dropout();
    buffer_lstm->disable_dropout();
    const_lstm_fwd.disable_dropout();
    const_lstm_rev.disable_dropout();
  }
  // variables in the computation graph representing the parameters
  Expression pbias = parameter(*hg, p_pbias);
  Expression S = parameter(*hg, p_S);
  Expression B = parameter(*hg, p_B);
  Expression A = parameter(*hg, p_A);
  Expression ptbias, ptW;
  // variables for edge labeling
  Expression Be, Se, eW, ebias, e2lbl, lbias;
  if (implicit_reduce) {
    ptbias = parameter(*hg, p_ptbias);
    ptW = parameter(*hg, p_ptW);
  }
  Expression p2w;
  if (use_pos) {
    p2w = parameter(*hg, p_p2w);
  }
  if (use_edges) {
    Be = parameter(*hg, p_Be);
    Se = parameter(*hg, p_Se);
    eW = parameter(*hg, p_eW);
    ebias = parameter(*hg, p_ebias);
    e2lbl = parameter(*hg, p_e2lbl);
    lbias = parameter(*hg, p_lbias);
  }
  Expression ib = parameter(*hg, p_ib);
  Expression cbias = parameter(*hg, p_cbias);
  Expression w2l = parameter(*hg, p_w2l);
  Expression t2l;
  if (p_t2l)
    t2l = parameter(*hg, p_t2l);
  Expression p2a = parameter(*hg, p_p2a);
  Expression abias = parameter(*hg, p_abias);
  Expression action_start = parameter(*hg, p_action_start);
  Expression cW = parameter(*hg, p_cW);

  action_lstm.add_input(action_start);

  vector<Expression> buffer(sent.size() + 1);  // variables representing word embeddings
  vector<int> bufferi(sent.size() + 1);  // position of the words in the sentence
  // precompute buffer representation from left to right

  // in the discriminative model, here we set up the buffer contents
  for (unsigned i = 0; i < sent.size(); ++i) {
    unsigned wordid = sent.raw[i]; // this will be equal to unk at dev/test
    if (is_evaluation) {
      wordid = sent.unk[i];
    } else if (build_training_graph && singletons.size() > wordid &&
        singletons[wordid] && rand01() > 0.5) {
      wordid = sent.unk[i];
    }
    Expression w = lookup(*hg, p_w, wordid);

    vector<Expression> args = {ib, w2l, w}; // learn embeddings
    if (p_t && pretrained->count(sent.lc[i])) {  // include fixed pretrained vectors?
      Expression t = const_lookup(*hg, p_t, sent.lc[i]);
      args.push_back(t2l);
      args.push_back(t);
    }
    if (use_pos) {
      args.push_back(p2w);
      Expression p = lookup(*hg, p_pos, sent.pos_val[sent.pos_offset[i]]);
      for (unsigned j = sent.pos_offset[i] + 1; j < sent.pos_offset[i+1]; j++) {
        p = p + lookup(*hg, p_pos, sent.pos_val[j]);
      }
      args.push_back(p);
    }
    buffer[sent.size() - i] = rectify(affine_transform(args));
    bufferi[sent.size() - i] = i;
  }
  // dummy symbol to represent the empty buffer
  buffer[0] = parameter(*hg, p_buffer_guard);
  bufferi[0] = -999;
  for (auto& b : buffer)
    buffer_lstm->add_input(b);

  vector<Expression> stack;  // variables representing subtree embeddings
  vector<int> stacki; // position of words in the sentence of head of subtree
  stack.push_back(parameter(*hg, p_stack_guard));
  stacki.push_back(-999); // not used for anything
  // drive dummy symbol on stack through LSTM
  stack_lstm.add_input(stack.back());
  vector<int> is_open_paren; // -1 if no nonterminal has a parenthesis open, otherwise index of NT
  is_open_paren.push_back(-1); // corresponds to dummy symbol
  vector<Expression> log_probs;
  string rootword;
  unsigned action_count = 0;  // incremented at each prediction
  unsigned nt_count = 0; // number of times an NT has been introduced
  unsigned label_count = 0; // number of assigned edge labels
  vector<unsigned> current_valid_actions;
  int nopen_parens = 0;
  char prev_a = '0';
  while(stack.size() > 2 || buffer.size() > 1) {
    // get list of possible actions for the current parser state
    current_valid_actions.clear();
    for (auto a: grammar->possible_actions) {
      if (IsActionForbidden_Discriminative(grammar->adict.Convert(a),
            prev_a, buffer.size(), stack.size(), nopen_parens,
            implicit_reduce))
        continue;
      current_valid_actions.push_back(a);
    }
    //cerr << "valid actions = " << current_valid_actions.size() << endl;

    // p_t = pbias + S * slstm + B * blstm + A * almst
    Expression stack_summary = stack_lstm.back();
    Expression action_summary = action_lstm.back();
    Expression buffer_summary = buffer_lstm->back();
    if (apply_dropout) {
      stack_summary = dropout(stack_summary, dropout_amount);
      action_summary = dropout(action_summary, dropout_amount);
      buffer_summary = dropout(buffer_summary, dropout_amount);
    }
    Expression p_t = affine_transform({pbias, S, stack_summary, B, buffer_summary, A, action_summary});
    Expression nlp_t = rectify(p_t);
    //if (build_training_graph) nlp_t = dropout(nlp_t, 0.4);
    // r_t = abias + p2a * nlp
    Expression r_t = affine_transform({abias, p2a, nlp_t});
    if (sample && alpha != 1.0f) r_t = r_t * alpha;
    // adist = log_softmax(r_t, current_valid_actions)
    Expression adiste = log_softmax(r_t, current_valid_actions);
    vector<float> adist = as_vector(hg->incremental_forward());
    double best_score = adist[current_valid_actions[0]];
    unsigned model_action = current_valid_actions[0];
    if (sample) {
      double p = rand01();
      assert(current_valid_actions.size() > 0);
      unsigned w = 0;
      for (; w < current_valid_actions.size(); ++w) {
        p -= exp(adist[current_valid_actions[w]]);
        if (p < 0.0) { break; }
      }
      if (w == current_valid_actions.size()) w--;
      model_action = current_valid_actions[w];
    } else { // max
      for (unsigned i = 1; i < current_valid_actions.size(); ++i) {
        if (adist[current_valid_actions[i]] > best_score) {
          best_score = adist[current_valid_actions[i]];
          model_action = current_valid_actions[i];
        }
      }
    }
    unsigned action = model_action;
    if (build_training_graph) {  // if we have reference actions (for training) use the reference action
      if (action_count >= correct_actions.size()) {
        cerr << "Correct action list exhausted, but not in final parser state.\n";
        abort();
      }
      action = correct_actions[action_count];
      if (model_action == action) { (*right)++; }
    } else {
      //cerr << "Chosen action: " << adict.Convert(action) << endl;
    }
    //cerr << "prob ="; for (unsigned i = 0; i < adist.size(); ++i) { cerr << ' ' << adict.Convert(i) << ':' << adist[i]; }
    //cerr << endl;
    ++action_count;
    log_probs.push_back(pick(adiste, action));
    results.push_back(action);

    // add current action to action LSTM
    Expression actione = lookup(*hg, p_a, action);
    action_lstm.add_input(actione);

    // do action
    const string& actionString=grammar->adict.Convert(action);
    //cerr << "ACT: " << actionString << endl;
    const char ac = actionString[0];
    const char ac2 = actionString[1];
    prev_a = ac;

    if (ac =='S' && ac2=='H') {  // SHIFT
      assert(buffer.size() > 1); // dummy symbol means > 1 (not >= 1)
      if (implicit_reduce) {
        --nopen_parens;
        int i = is_open_paren.size() - 1;
        assert(is_open_paren[i] >= 0);
        Expression nonterminal = lookup(*hg, p_ntup, is_open_paren[i]);
        Expression terminal = buffer.back();
        Expression c = concatenate({nonterminal, terminal});
        Expression pt = rectify(affine_transform({ptbias, ptW, c}));
        stack.pop_back();
        stacki.pop_back();
        stack_lstm.rewind_one_step();
        buffer.pop_back();
        bufferi.pop_back();
        buffer_lstm->rewind_one_step();
        is_open_paren.pop_back();
        stack_lstm.add_input(pt);
        stack.push_back(pt);
        stacki.push_back(999);
        is_open_paren.push_back(-1);
      } else {
        stack.push_back(buffer.back());
        stack_lstm.add_input(buffer.back());
        stacki.push_back(bufferi.back());
        buffer.pop_back();
        buffer_lstm->rewind_one_step();
        bufferi.pop_back();
        is_open_paren.push_back(-1);
      }
    } else if (ac == 'N') { // NT
      ++nopen_parens;
      assert(buffer.size() > 1);
      auto it = grammar->action2NTindex.find(action);
      assert(it != grammar->action2NTindex.end());
      int nt_index = it->second;
      nt_count++;
      Expression nt_embedding = lookup(*hg, p_nt, nt_index);
      stack.push_back(nt_embedding);
      stack_lstm.add_input(nt_embedding);
      stacki.push_back(-1);
      is_open_paren.push_back(nt_index);
    } else { // REDUCE
      --nopen_parens;
      assert(stack.size() > 2); // dummy symbol means > 2 (not >= 2)
      // find what paren we are closing
      int i = is_open_paren.size() - 1;
      while(is_open_paren[i] < 0) { --i; assert(i >= 0); }
      Expression nonterminal = lookup(*hg, p_ntup, is_open_paren[i]);
      int nchildren = is_open_paren.size() - i - 1;
      assert(nchildren > 0);
      //cerr << "  number of children to reduce: " << nchildren << endl;
      vector<Expression> children(nchildren);
      const_lstm_fwd.start_new_sequence();
      const_lstm_rev.start_new_sequence();

      // REMOVE EVERYTHING FROM THE STACK THAT IS GOING
      // TO BE COMPOSED INTO A TREE EMBEDDING
      is_open_paren.pop_back(); // nt symbol
      stacki.pop_back(); // nonterminal dummy
      stack.pop_back(); // nonterminal dummy
      stack_lstm.rewind_one_step(); // nt symbol
      for (i = 0; i < nchildren; ++i) {
        children[i] = stack.back();
        stacki.pop_back();
        stack.pop_back();
        stack_lstm.rewind_one_step();
        is_open_paren.pop_back();
      }

      // BUILD TREE EMBEDDING USING BIDIR LSTM
      vector<Expression> fwd_hidden(nchildren);
      vector<Expression> rev_hidden(nchildren);
      const_lstm_fwd.add_input(nonterminal);
      const_lstm_rev.add_input(nonterminal);
      for (i = 0; i < nchildren; ++i) {
        const_lstm_fwd.add_input(children[i]);
        const_lstm_rev.add_input(children[nchildren - i - 1]);
        if (predict_edges || train_edges) {
          fwd_hidden[i] = const_lstm_fwd.back();
          rev_hidden[nchildren - i - 1] = const_lstm_rev.back();
          if (apply_dropout) {
            fwd_hidden[i] = dropout(fwd_hidden[i], dropout_amount);
            rev_hidden[nchildren - i - 1] =
              dropout(rev_hidden[nchildren - i -1], dropout_amount);
          }
        }
      }
      if (predict_edges || train_edges) {
        Expression e_sb = affine_transform({ebias,
            Se, stack_summary, Be, buffer_summary});
        for (i = 0; i < nchildren; i++) {
          Expression e_lstm = concatenate({fwd_hidden[i], rev_hidden[i]});
          Expression compose = rectify(affine_transform({e_sb, eW, e_lstm}));
          Expression r_lbl = affine_transform({lbias, e2lbl, compose});
          Expression ldiste = log_softmax(r_lbl);
          vector<float> ldist = as_vector(hg->incremental_forward());
          double best_labelscore = ldist[0];
          unsigned model_label = 0;
          if (sample) {
            double p = rand01();
            unsigned w = 0;
            for (; w < grammar->edgedict.size(); ++w) {
              p -= exp(ldist[w]);
              if (p < 0.0) break;
            }
            if (w == grammar->edgedict.size()) --w;
            model_label = w;
          } else { // argmax
            for (unsigned i = 1; i < grammar->edgedict.size(); i++) {
              if (ldist[i] > best_labelscore) {
                best_labelscore = ldist[i];
                model_label = i;
              }
            }
          }
          unsigned elabel = model_label;
          if (build_training_graph && train_edges) {
            if (label_count >= correct_labels.size()) {
              cerr << "Correct label list exhausted, but need a label.\n";
              abort();
            }
            elabel = correct_labels[label_count];
          }
          ++label_count;
          if (predict_edges) {
            result_labels->push_back(elabel);
          }
          log_probs.push_back(pick(ldiste, elabel));
        }
      }
      Expression cfwd = const_lstm_fwd.back();
      Expression crev = const_lstm_rev.back();
      if (apply_dropout) {
        cfwd = dropout(cfwd, dropout_amount);
        crev = dropout(crev, dropout_amount);
      }
      Expression c = concatenate({cfwd, crev});
      Expression composed = rectify(affine_transform({cbias, cW, c}));
      stack_lstm.add_input(composed);
      stack.push_back(composed);
      stacki.push_back(999); // who knows, should get rid of this
      is_open_paren.push_back(-1); // we just closed a paren at this position
    }
  }
  if (build_training_graph && action_count != correct_actions.size()) {
    cerr << "Unexecuted actions remain but final state reached!\n";
    abort();
  }
  if (train_edges && label_count != correct_labels.size()) {
    cerr << "Unused labels remain but final state reached\n";
    abort();
  }
  assert(stack.size() == 2); // guard symbol, root
  assert(stacki.size() == 2);
  assert(buffer.size() == 1); // guard symbol
  assert(bufferi.size() == 1);
  Expression tot_neglogprob = -sum(log_probs);
  assert(tot_neglogprob.pg != nullptr);
  return results;
}

void load_model(Model &model, const string &fname) {
  ifstream in(fname);
  boost::archive::text_iarchive ia(in);
  ia >> model;
}

void save_model(Model &model, const string &fname) {
  ofstream out(fname);
  boost::archive::text_oarchive oa(out);
  oa << model;
}
