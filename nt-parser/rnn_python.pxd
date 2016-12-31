import sys
from libc.stdlib cimport malloc, free
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp cimport bool

cdef extern from "cnn/init.h" namespace "cnn":
    cdef void Initialize(int& argc, char **&argv, unsigned random_seed)

cdef extern from "cnn/dict.h" namespace "cnn":
    cdef cppclass Dict:
        int Convert(string)
        string Convert(int)

cdef extern from "cnn/tensor.h" namespace "cnn":
    cdef cppclass Tensor:
        pass
    float as_scalar(Tensor&)
    vector[float] as_vector(Tensor&)

cdef extern from "cnn/cnn.h" namespace "cnn":
    cdef cppclass ComputationGraph:
        __init__()
        Tensor& incremental_forward()

cdef extern from "cnn/training.h" namespace "cnn":
    cdef cppclass c_Model "Model":
        __init__()


ctypedef vector[unsigned] vec_unsigned

cdef extern from "nt-parser/rnn_grammar.h":
    cdef cppclass c_Sentence "Sentence":
        vector[int] raw
        vector[int] unk
        vector[int] lc
        vector[int] pos_val
        vector[int] pos_offset
        int size()
    cdef cppclass c_RNNGrammar "RNNGrammar"
    cdef cppclass Corpus:
        c_RNNGrammar *grammar
        __init__(c_RNNGrammar *)
        int size()
        vector[c_Sentence] sents
        vector[vec_unsigned] actions
        vector[vec_unsigned] edge_labels
        void load_oracle(string fname)
    cdef cppclass c_RNNGrammar "RNNGrammar":
        Dict termdict
        Dict adict
        Dict ntermdict
        Dict posdict
        Dict edgedict
        Corpus *load_corpus(string fname, bool split)
        void Freeze()
        void save_to_files(string prefix)

cdef extern from "nt-parser/pretrained.h" namespace "parser":
    void ReadEmbeddings_word2vec(
        string fname, Dict *dict,
        unordered_map[unsigned, vector[float]] *pretrained)


cdef extern from "nt-parser/parser_disc.h":
    cdef cppclass c_NetworkSettings "NetworkSettings":
        unsigned LAYERS
        unsigned INPUT_DIM
        unsigned HIDDEN_DIM
        unsigned ACTION_DIM
        unsigned PRETRAINED_DIM
        unsigned LSTM_INPUT_DIM
        unsigned POS_DIM
        float DROPOUT
        unsigned POS_SIZE
        bool USE_POS
        bool USE_EDGES
        unsigned IMPLICIT_REDUCE_AFTER_SHIFT
        float ALPHA
        unsigned compat_version
    cdef cppclass c_ParserBuilder "ParserBuilder":
        c_RNNGrammar *grammar
        float alpha
        __init__(c_RNNGrammar *, c_NetworkSettings *,
                 vector[bool],
                 c_Model *, unordered_map[unsigned, vector[float]] *)
        vector[unsigned] log_prob_parser(ComputationGraph *,
                                         c_Sentence,
                                         vector[unsigned],
                                         vector[unsigned],
                                         vector[unsigned] *,
                                         double *,
                                         bool, bool)
    cdef void load_model(c_Model, string)
    cdef void save_model(c_Model, string)

cdef class RNNGrammar:
    cdef c_RNNGrammar gram
    cdef bool split
    cpdef RNNG_Corpus load_corpus(self, fname)

cdef class RNNG_Corpus:
    cdef Corpus *corpus
    cdef readonly RNNGrammar grammar

cdef class RNNG_Parser:
    cdef RNNGrammar grammar
    cdef c_ParserBuilder *parser
    cdef c_Model *model
    cdef unordered_map[unsigned, vector[float]] pretrained
    cdef bint use_morph
    cdef readonly object pipelines
    cdef readonly object features
    cdef readonly object parens

cdef class SampleWrapper:
    cdef readonly RNNG_Parser parser
    cdef public int k
    cdef public float alpha

cdef init(argv=*, int random_seed=*)
