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

string NetworkSettingsGen::propose_filename() {
  ostringstream os;
  os << "ntparse_gen"
     << "_D" << DROPOUT
     << '_' << LAYERS
     << '_' << INPUT_DIM
     << '_' << HIDDEN_DIM
     << '_' << ACTION_DIM
     << '_' << LSTM_INPUT_DIM
     << "-pid" << getpid() << ".params";
  return os.str();
}

ParserBuilderGen::ParserBuilderGen(
        const RNNGrammar *gram,
        ClassFactoredSoftmaxBuilder *cfsm_init,
        const NetworkSettings *settings,
        Model* model):
    grammar(gram),
    cfsm(cfsm_init),
    nt_size(gram->ntermdict.size()),
    action_size(gram->adict.size()),
    vocab_size(gram->termdict.size()),
    pos_size(gram->posdict.size()),
    dropout_amount(settings->DROPOUT),
    pretrained(pre),
    singletons(singleton_init),
    // summaries of the stack and of previous actions
    stack_lstm(settings->LAYERS, settings->LSTM_INPUT_DIM,
            settings->HIDDEN_DIM, model),
    term_lstm(settings->LAYERS, settings->ACTION_DIM,
            settings->HIDDEN_DIM, model),
    action_lstm(settings->LAYERS, settings->ACTION_DIM,
            settings->HIDDEN_DIM, model),
    // used to compose children (fwd/rev) into a representation of the node
    const_lstm_fwd(settings->LAYERS, settings->LSTM_INPUT_DIM,
            settings->LSTM_INPUT_DIM, model),
    const_lstm_rev(settings->LAYERS, settings->LSTM_INPUT_DIM,
            settings->LSTM_INPUT_DIM, model),
    p_w(model->add_lookup_parameters(vocab_size, {settings->INPUT_DIM})),
    p_t(model->add_lookup_parameters(vocab_size, {settings->INPUT_DIM})),
    p_nt(model->add_lookup_parameters(nt_size, {settings->LSTM_INPUT_DIM})),
    p_ntup(model->add_lookup_parameters(nt_size, {settings->LSTM_INPUT_DIM})),
    p_a(model->add_lookup_parameters(action_size, {settings->ACTION_DIM})),
    p_pbias(model->add_parameters({settings->HIDDEN_DIM})),
    p_A(model->add_parameters({settings->HIDDEN_DIM, settings->HIDDEN_DIM})),
    p_S(model->add_parameters({settings->HIDDEN_DIM, settings->HIDDEN_DIM})),
    p_T(model->add_parameters({settings->HIDDEN_DIM, settings->HIDDEN_DIM})),
    p_cbias(model->add_parameters({settings->LSTM_INPUT_DIM})),
    p_p2a(model->add_parameters({action_size, settings->HIDDEN_DIM})),
    p_action_start(model->add_parameters({settings->ACTION_DIM})),
    p_abias(model->add_parameters({action_size})),

    p_stack_guard(model->add_parameters({settings->LSTM_INPUT_DIM})),

    p_cW(model->add_parameters({settings->LSTM_INPUT_DIM,
                settings->LSTM_INPUT_DIM * 2}))
{
  // everything all set
}

bool IsActionForbidden_Generative(const string& a, char prev_a,
        unsigned tsize, unsigned ssize, unsigned nopen_parens)
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

    // you can't reduce after an NT action
    if (is_reduce && prev_a == 'N') return true;
    return false;
}

// *** if correct_actions is empty, this runs greedy decoding ***
// returns parse actions for input sentence (in training just returns the reference)
// this lets us use pretrained embeddings, when available, for words that were OOV in the
// parser training data
// set sample=true to sample rather than max
vector<unsigned> ParserBuilderGen::log_prob_parser(ComputationGraph* hg,
        const Sentence& sent,
        const vector<unsigned>& correct_actions,
        double *right,
        bool is_evaluation) {
    vector<unsigned> results;
    vector<string> stack_content;
    stack_content.push_back("ROOT_GUARD");
    const bool sample = sent.size() == 0;
    const bool build_training_graph = correct_actions.size() > 0;
    assert(sample || build_training_graph);
    bool apply_dropout = (dropout_amount && !is_evaluation);
    if (sample) apply_dropout = false;

    if (apply_dropout) {
        stack_lstm.set_dropout(dropout_amount);
        term_lstm.set_dropout(dropout_amount);
        action_lstm.set_dropout(dropout_amount);
        const_lstm_fwd.set_dropout(dropout_amount);
        const_lstm_rev.set_dropout(dropout_amount);
    } else {
        stack_lstm.disable_dropout();
        term_lstm->disable_dropout();
        action_lstm.disable_dropout();
        const_lstm_fwd.disable_dropout();
        const_lstm_rev.disable_dropout();
    }
    term_lstm.new_graph(*hg);
    stack_lstm.new_graph(*hg);
    action_lstm.new_graph(*hg);
    const_lstm_fwd.new_graph(*hg);
    const_lstm_rev.new_graph(*hg);
    cfsm->new_graph(*hg);
    term_lstm.start_new_sequence();
    stack_lstm.start_new_sequence();
    action_lstm.start_new_sequence();
    // variables in the computation graph representing the parameters
    Expression pbias = parameter(*hg, p_pbias);
    Expression S = parameter(*hg, p_S);
    Expression A = parameter(*hg, p_A);
    Expression T = parameter(*hg, p_T);
    Expression cbias = parameter(*hg, p_cbias);
    Expression p2a = parameter(*hg, p_p2a);
    Expression abias = parameter(*hg, p_abias);
    Expression action_start = parameter(*hg, p_action_start);
    Expression cW = parameter(*hg, p_cW);

    action_lstm.add_input(action_start);

    vector<Expression> terms(1, lookup(*hg, p_w, kSOS));
    term_lstm.add_input(stack.back());

    vector<Expression> stack;  // variables representing subtree embeddings
    stack.push_back(parameter(*hg, p_stack_guard));
    // drive dummy symbol on stack through LSTM
    stack_lstm.add_input(stack.back());
    vector<int> is_open_paren; // -1 if no nonterminal has a parenthesis open, otherwise index of NT
    is_open_paren.push_back(-1); // corresponds to dummy symbol
    vector<Expression> log_probs;
    string rootword;
    unsigned action_count = 0;  // incremented at each prediction
    unsigned nt_count = 0; // number of times an NT has been introduced
    vector<unsigned> current_valid_actions;
    int nopen_parens = 0;
    char prev_a = '0';
    unsigned termc = 0;
    while(stack.size() > 2 || buffer.size() > 1) {
        // get list of possible actions for the current parser state
        current_valid_actions.clear();
        for (auto a: grammar->possible_actions) {
            if (IsActionForbidden_Generative(grammar->adict.Convert(a),
                        prev_a, buffer.size(), stack.size(), nopen_parens))
                continue;
            current_valid_actions.push_back(a);
        }
        //cerr << "valid actions = " << current_valid_actions.size() << endl;

        // p_t = pbias + S * slstm + B * blstm + A * almst
        Expression stack_summary = stack_lstm.back();
        Expression action_summary = action_lstm.back();
        Expression term_summary = term_lstm->back();
        if (apply_dropout) {
            stack_summary = dropout(stack_summary, dropout_amount);
            action_summary = dropout(action_summary, dropout_amount);
            term_summary = dropout(term_summary, dropout_amount);
        }
        Expression p_t = affine_transform({pbias, S, stack_summary, B, buffer_summary, A, action_summary});
        Expression nlp_t = rectify(p_t);
        //if (build_training_graph) nlp_t = dropout(nlp_t, 0.4);
        // r_t = abias + p2a * nlp
        Expression r_t = affine_transform({abias, p2a, nlp_t});
        // adist = log_softmax(r_t, current_valid_actions)
        Expression adiste = log_softmax(r_t, current_valid_actions);
        unsigned action = 0;
        if (sample) {
          auto dist = as_vector(hg->incremental_forward());
          double p = rand01();
          assert(current_valid_actions.size() > 0);
          unsigned w = 0;
          for (; w < current_valid_actions.size(); ++w) {
            p -= exp(adist[current_valid_actions[w]]);
            if (p < 0.0) { break; }
          }
          if (w == current_valid_actions.size()) w--;
          action = current_valid_actions[w];
          const string& a = adict.Convert(action);
          if (a[0] == 'R') cerr << ")";
          if (a[0] == 'N') {
            int nt = grammar->action2NTindex[action];
            cerr << " (" << grammar->ntermdict.convert(nt);
          }
        } else {
          if (action_count >= correct_actions.size()) {
            cerr << "Correct action list exhausted, but not in final parser state.\n";
            abort()
          }
          action = correct_actions[action_count];
          ++action_count;
          lob_probs.push_back(pick(adiste, action));
        } 
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
          unsigned wordid = 0;
          if (sample) {
            wordid = cfsm->sample(nlp_t);
            cerr << " " << grammar->termdict.Convert(wordid);
          } else {
            assert(termc < sent.size());
            wordid = sent.raw[termc];
            log_probs.push_back(-cfsm->neg_log_softmax(nlp_t, wordid));
          }
          assert(wordid != 0);
          stack_content.push_back(grammar->termdict.Convert(wordid));
          ++ termc;
          Expression word = lookup(*hg, p_w, wordid);
          terms.push_back(word);
          term_lstm.add_input(word);
          stack.push_back(word);
          stack_lstm.add_input(word);
          is_open_paren.push_back(-1);
        } else if (ac == 'N') { // NT
            ++nopen_parens;
            assert(buffer.size() > 1);
            auto it = grammar->action2NTindex.find(action);
            assert(it != grammar->action2NTindex.end());
            int nt_index = it->second;
            nt_count++;
            stack_content.push_back(grammar->ntermdict.Convert(nt_index));
            Expression nt_embedding = lookup(*hg, p_nt, nt_index);
            stack.push_back(nt_embedding);
            stack_lstm.add_input(nt_embedding);
            is_open_paren.push_back(nt_index);
        } else { // REDUCE
            --nopen_parens;
            assert(stack.size() > 2); // dummy symbol means > 2 (not >= 2)
            assert(stack_content.size() > 2 && stack.size() == stack_content.size());
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
            string curr_word;
            for (i = 0; i < nchildren; ++i) {
              assert(stack_content.size() == stack.size());
              children[i] = stack.back();
              stack.pop_back();
              stack_lstm.rewind_one_step();
              is_open_paren.pop_back();
              stack_content.pop_back();
            }
            assert(stack_content.size() == stack.size());
            is_open_paren.pop_back(); // nt symbol
            stack.pop_back(); // nonterminal dummy
            stack_lstm.rewind_one_step(); // nt symbol
            curr_word = stack_content.back();
            stack_content.pop_back();
            assert(stack.size()==stack_content.size());

            // BUILD TREE EMBEDDING USING BIDIR LSTM
            const_lstm_fwd.add_input(nonterminal);
            const_lstm_rev.add_input(nonterminal);
            for (i = 0; i < nchildren; ++i) {
                const_lstm_fwd.add_input(children[i]);
                const_lstm_rev.add_input(children[nchildren - i - 1]);
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
            is_open_paren.push_back(-1); // we just closed a paren at this position
        }
    }
    if (action_count != correct_actions.size()) {
        cerr << "Unexecuted actions remain but final state reached!\n";
        abort();
    }
    assert(stack.size() == 2); // guard symbol, root
    if (!sample) {
      Expression tot_neglogprob = -sum(log_probs);
      assert(tot_neglogprob.pg != nullptr);
    }
    if (sample) cerr << "\n";
    return results;
}
