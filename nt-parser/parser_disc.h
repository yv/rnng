#include <unordered_map>
#include <vector>
#include <string>
#include "cnn/cnn.h"
#include "cnn/lstm.h"
#include "cnn/rnn.h"
#include "cnn/training.h"

using std::unordered_map;
using std::vector;
using std::string;
using cnn::LSTMBuilder;
using cnn::Parameters;
using cnn::LookupParameters;
using cnn::ComputationGraph;
using cnn::Model;

struct NetworkSettings {
    unsigned LAYERS = 2;
    unsigned INPUT_DIM = 40;
    unsigned HIDDEN_DIM = 60;
    unsigned ACTION_DIM = 36;
    unsigned PRETRAINED_DIM = 50;
    unsigned LSTM_INPUT_DIM = 60;
    // in discriminative parser, incorporate POS information in token embedding
    unsigned POS_DIM = 10;
    float DROPOUT = 0.0f;
    bool USE_POS = false;  
    bool SPLIT_POS = false;
    bool USE_EDGES = false;
    unsigned IMPLICIT_REDUCE_AFTER_SHIFT = 0;
    bool AVG_FEATURES = false;
    float ALPHA = 1.f;
    // 0 = initial buggy version; 1 = fixed version
    int compat_version = 0;
    string propose_filename();
};

struct ParserBuilder {
    const RNNGrammar *grammar;
    const unsigned nt_size;
    const unsigned action_size;
    const unsigned vocab_size;
    const unsigned pretrain_size;
    const unsigned pos_size;
    const unsigned edges_size;
    const float dropout_amount;
    const int compat_version;
    float alpha;
    const bool implicit_reduce;
    const bool use_pos;
    const bool use_edges;
    const bool avg_features;
    vector<bool> singletons;
    LSTMBuilder stack_lstm; // (layers, input, hidden, trainer)
    LSTMBuilder *buffer_lstm;
    LSTMBuilder action_lstm;
    LSTMBuilder const_lstm_fwd;
    LSTMBuilder const_lstm_rev;
    LookupParameters* p_w; // word embeddings
    LookupParameters* p_t; // pretrained word embeddings (not updated)
    LookupParameters* p_nt; // nonterminal embeddings
    LookupParameters* p_ntup; // nonterminal embeddings when used in a composed representation
    LookupParameters* p_a; // input action embeddings
    LookupParameters* p_pos; // pos embeddings (optional)
    Parameters* p_p2w;  // pos2word mapping (optional)
    Parameters* p_ptbias; // preterminal bias (used with IMPLICIT_REDUCE_AFTER_SHIFT)
    Parameters* p_ptW;    // preterminal W (used with IMPLICIT_REDUCE_AFTER_SHIFT)
    Parameters* p_pbias; // parser state bias
    Parameters* p_A; // action lstm to parser state
    Parameters* p_B; // buffer lstm to parser state
    Parameters* p_S; // stack lstm to parser state
    Parameters* p_w2l; // word to LSTM input
    Parameters* p_t2l; // pretrained word embeddings to LSTM input
    Parameters* p_ib; // LSTM input bias
    Parameters* p_cbias; // composition function bias
    Parameters* p_p2a;   // parser state to action
    Parameters* p_action_start;  // action bias
    Parameters* p_abias;  // action bias
    Parameters* p_buffer_guard;  // end of buffer
    Parameters* p_stack_guard;  // end of stack

    Parameters* p_cW;
    // for edge encoder
    Parameters * p_Be;
    Parameters * p_Se;
    Parameters * p_eW;
    Parameters * p_ebias;
    Parameters * p_e2lbl;
    Parameters * p_lbias;

    unordered_map<unsigned, vector<float>> *pretrained;

    explicit ParserBuilder(const RNNGrammar *gram,
        const NetworkSettings *settings,
        const vector<bool> &singleton_init,
        Model* model,
        unordered_map<unsigned, vector<float>>* pre);
    vector<Expression> prepare_buffer(
        ComputationGraph *hg,
        const Sentence& sent,
        bool is_evaluation);
    vector<unsigned> log_prob_parser(ComputationGraph* hg,
        const Sentence& sent,
        const vector<unsigned>& correct_actions,
        const vector<unsigned>& correct_labels,
        vector<unsigned> *result_labels,
        double *right,
        bool is_evaluation,
        bool sample = false);
    unsigned get_mfeat_category(int mfeat) {
      const string& s=grammar->posdict.Convert(mfeat);
      char s0 = s[0];
      if (s0 >= '0' && s0 <= '9') {
        return s0 - '0';
      } else {
        return 0;
      }
    }

};


bool IsActionForbidden_Discriminative(const string& a, char prev_a,
        unsigned bsize, unsigned ssize, unsigned nopen_parens);

void load_model(Model &model, const string &fname);
void save_model(Model &model, const string &fname);
