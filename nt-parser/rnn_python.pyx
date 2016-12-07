#
# RNN grammar python
#
import sys
from pytree.tree import TerminalNode, NontermNode
from itertools import izip

cdef class RNNGrammar:
    def __init__(self, split=True):
        self.split = split
    cpdef RNNG_Corpus load_corpus(self, fname):
        cdef Corpus *corpus
        cdef RNNG_Corpus result
        corpus = self.gram.load_corpus(fname, self.split)
        result = RNNG_Corpus()
        result.corpus = corpus
        result.grammar = self
        return result
    def Freeze(self):
        self.gram.Freeze()

cdef string kSHIFT = 'SHIFT'
cdef string kREDUCE = 'REDUCE'

cdef object build_tree(c_Sentence *sent, vector[unsigned] *actions,
                       vector[unsigned] *labels,
                       c_RNNGrammar *grammar, object words=None):
    cdef int label_idx = 0
    cdef int term_idx = 0
    terms = []
    nt_idx = []
    for i from 0 <= i < sent.size():
        cat = grammar.posdict.Convert(sent.pos_val[sent.pos_offset[i]])
        word = grammar.termdict.Convert(sent.unk[i])
        term = TerminalNode(cat, word)
        if words is None:
            term.word_orig = grammar.termdict.Convert(sent.raw[i])
        else:
            term.word_orig = words[i]
        terms.append(term)
    if actions.size() > 0:
        stack = []
        nt_idx = []
        nt_cat = []
        term_idx = 0
        for i from 0 <= i < actions.size():
            act = grammar.adict.Convert(actions[0][i])
            if act == kSHIFT:
                stack.append(terms[term_idx])
                term_idx += 1
            elif act == kREDUCE:
                start = nt_idx.pop()
                cat = nt_cat.pop()
                chlds = stack[start:]
                del stack[start:]
                nt = NontermNode(cat)
                nt.children = chlds
                for chld in chlds:
                    chld.parent = nt
                if labels != NULL and labels.size() > 0:
                    for j, chld in izip(
                        range(label_idx, label_idx + len(chlds)),
                        chlds):
                        chld.edge_label = grammar.edgedict.Convert(labels[0][j])
                    label_idx += len(chlds)
                stack.append(nt)
            elif act.startswith('NT('):
                act_py = str(act)
                nt_idx.append(len(stack))
                nt_cat.append(act_py[3:-1])
        assert len(stack) == 1
        nt = stack[0]
    else:
        nt = NontermNode('ROOT')
        nt.children = terms
        for term in terms:
            term.parent = nt
    return nt

cdef class RNNG_Corpus:
    def __dealloc__(self):
        if self.corpus != NULL:
            del self.corpus
            self.corpus = NULL
    def __len__(self):
        return self.corpus.sents.size()
    def __getitem__(self, idx):
        assert idx >= 0
        assert self.corpus != NULL
        assert idx < self.corpus.size()
        cdef int i, term_idx
        cdef c_Sentence *sent = &self.corpus.sents[idx]
        cdef vector[unsigned] *actions = &self.corpus.actions[idx]
        cdef vector[unsigned] *edges = &self.corpus.edge_labels[idx]
        nt = build_tree(sent, actions, edges, self.corpus.grammar)
        return nt

cdef class RNNG_Parser:
    def __init__(self, fname_oracle, fname_model, fname_embeddings,
                 pipelines, use_morph, use_edges):
        cdef c_NetworkSettings param
        cdef vector[bool] single
        # actual network parameters are filled from the serialized model
        # but the model needs to be initialized with suitable values
        param.LAYERS = 2
        param.INPUT_DIM = 40
        param.HIDDEN_DIM = 60
        param.ACTION_DIM = 36
        param.PRETRAINED_DIM = 100
        param.LSTM_INPUT_DIM = 60
        param.POS_DIM = 10
        param.USE_POS = True
        param.USE_EDGES = use_edges
        self.use_morph = use_morph
        self.grammar = RNNGrammar(use_morph)
        self.grammar.load_corpus(fname_oracle)
        ReadEmbeddings_word2vec(
            fname_embeddings,
            &self.grammar.gram.termdict,
            &self.pretrained)
        self.grammar.Freeze()
        self.model = new c_Model()
        self.parser = new c_ParserBuilder(
            &self.grammar.gram,
            &param, single, self.model,
            &self.pretrained)
        load_model(self.model[0], fname_model)
        self.pipelines = pipelines
    def do_parse(self, line_in, preproc):
        cdef c_Sentence sent
        cdef string s_w, s_wl, s_tag, s_wu
        cdef int id_w, id_wl, id_wu, id_tag
        cdef vector[unsigned] actions
        cdef vector[unsigned] labels_in
        cdef vector[unsigned] labels_result
        cdef vector[unsigned] actions_in
        cdef ComputationGraph graph
        cdef double right
        tags = preproc['pos']
        morph = preproc['morph']
        words = line_in.split(' ')
        line_out = line_in
        for pipeline in self.pipelines:
            line_out = pipeline.process_line(line_out)
            line_out = line_out.rstrip('\r\n')
        line_out = line_out.replace('(', '_RRB_')
        line_out = line_out.replace(')', '_LRB_')
        words_unk = line_out.split()
        sent.pos_offset.push_back(0)
        for i, w in enumerate(words):
            tag = tags[i]
            if self.use_morph:
                morphtag = morph[i]
            else:
                morphtag = "_"
            wu = words_unk[i]
            w_lower = w.decode('UTF-8').lower().encode('UTF-8')
            s_w = w
            s_wl = w_lower
            s_wu = wu
            s_tag = tag
            id_w = self.grammar.gram.termdict.Convert(s_w)
            id_wl = self.grammar.gram.termdict.Convert(s_wl)
            id_wu = self.grammar.gram.termdict.Convert(s_wu)
            id_tag = self.grammar.gram.posdict.Convert(s_tag)
            sent.raw.push_back(id_w)
            sent.lc.push_back(id_wl)
            sent.unk.push_back(id_wu)
            sent.pos_val.push_back(id_tag)
            if self.use_morph and morphtag != '_':
                for morphpart in morphtag.split('|'):
                    s_tag = morphpart
                    id_tag = self.grammar.gram.posdict.Convert(s_tag)
            sent.pos_offset.push_back(sent.pos_val.size())
        actions = self.parser.log_prob_parser(
            &graph, sent, actions_in, labels_in,
            &labels_result, &right, True, False)
        nt = build_tree(&sent, &actions, &labels_result,
                        &self.grammar.gram, words)
        return nt
    def do_sample(self, line_in, preproc, n_samples=100, alpha=0.8):
        cdef c_Sentence sent
        cdef string s_w, s_wl, s_tag, s_wu
        cdef int id_w, id_wl, id_wu, id_tag
        cdef vector[unsigned] actions
        cdef vector[unsigned] actions_in
        cdef vector[unsigned] labels_result
        cdef vector[unsigned] labels_in
        cdef ComputationGraph graph
        cdef double right
        tags = preproc['pos']
        morphtags = preproc['morph']
        words = line_in.split(' ')
        line_out = line_in
        for pipeline in self.pipelines:
            line_out = pipeline.process_line(line_out)
            line_out = line_out.rstrip('\r\n')
        line_out = line_out.replace('(', '_RRB_')
        line_out = line_out.replace(')', '_LRB_')
        words_unk = line_out.split()
        for w, tag, morphtag, wu in izip(words, tags, morphtags, words_unk):
            w_lower = w.decode('UTF-8').lower().encode('UTF-8')
            s_w = w
            s_wl = w_lower
            s_wu = wu
            s_tag = tag
            id_w = self.grammar.gram.termdict.Convert(s_w)
            id_wl = self.grammar.gram.termdict.Convert(s_wl)
            id_wu = self.grammar.gram.termdict.Convert(s_wu)
            id_tag = self.grammar.gram.posdict.Convert(s_tag)
            sent.raw.push_back(id_w)
            sent.lc.push_back(id_wl)
            sent.unk.push_back(id_wu)
            sent.pos_val.push_back(id_tag)
            if self.use_morph and morphtag != '_':
                for morphpart in morphtag.split('|'):
                    s_tag = morphpart
                    id_tag = self.grammar.gram.posdict.Convert(s_tag)
            sent.pos_offset.push_back(sent.pos_val.size())
        parse_result = []
        for i in xrange(n_samples):
            labels_result.clear()
            actions = self.parser.log_prob_parser(
                &graph, sent, actions_in, labels_in,
                &labels_result, &right, True, True)
            score = as_scalar(graph.incremental_forward())
            nt = build_tree(&sent, &actions, &labels_result,
                            &self.grammar.gram, words)
            parse_result.append((score, nt))
        parse_result.sort(reverse=True)
        return parse_result

cdef class SampleWrapper:
    def __init__(self, parser, k=100, alpha=0.8):
        self.parser = parser
        self.k = k
        self.alpha = alpha
    def do_parse(self, line_in, preproc):
        return self.parser.do_sample(line_in, preproc,
                                     self.k, self.alpha)

cdef init(argv=None, int random_seed=0):
    if argv is None:
        argv = sys.argv
    cdef int argc = len(argv)
    cdef char **c_argv
    args = [bytes(x) for x in argv]
    c_argv = <char **>malloc(sizeof(char *) * len(args))
    for idx, s in enumerate(args):
        c_argv[idx] = s
    Initialize(argc, c_argv, random_seed)
    free(c_argv)

init(['nt-parser', '--cnn-mem', '1700'])

