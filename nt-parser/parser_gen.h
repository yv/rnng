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

struct NetworkSettingsGen {
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
    unsigned IMPLICIT_REDUCE_AFTER_SHIFT = 0;
    float ALPHA = 1.f;
    string propose_filename();
};


struct ParserBuilderGen {
    const RNNGrammar *grammar;
    ClassFactoredSoftmaxBuilder *cfsm;
    const unsigned nt_size;
    const unsigned action_size;
    const unsigned vocab_size;
    const unsigned pos_size;
    const float dropout_amount;
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
    Parameters* p_pbias; // parser state bias
    Parameters* p_A; // action lstm to parser state
    Parameters* p_S; // stack lstm to parser state
    Parameters* p_T; // term lstm to parser state
    Parameters* p_w2l; // word to LSTM input
    Parameters* p_t2l; // pretrained word embeddings to LSTM input
    Parameters* p_ib; // LSTM input bias
    Parameters* p_cbias; // composition function bias
    Parameters* p_p2a;   // parser state to action
    Parameters* p_action_start;  // action bias
    Parameters* p_abias;  // action bias
    Parameters* p_stack_guard;  // end of stack

    Parameters* p_cW;

    explicit ParserBuilderGen(const RNNGrammar *gram,
        ClassFactoredSoftmaxBuilder *cfsm_init,
        const NetworkSettings *settings,
        Model* model);
    vector<unsigned> log_prob_parser(ComputationGraph* hg,
        const Sentence& sent,
        const vector<unsigned>& correct_actions,
        double *right,
        bool is_evaluation,
        bool sample = false);
};


bool IsActionForbidden_Generative(const string& a, char prev_a,
    unsigned bsize, unsigned ssize, unsigned nopen_parens);
