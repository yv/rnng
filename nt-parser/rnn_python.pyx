#
# RNN grammar python
#
import sys
import regex
import os.path
from pytree.tree import TerminalNode, NontermNode
from itertools import izip

NEEDED_MEM = 1700
DONE_INIT = False

cdef class RNNGrammar:
    def __init__(self, split=True):
        self.split = split
    cpdef RNNG_Corpus load_corpus(self, fname):
        cdef Corpus *corpus
        cdef RNNG_Corpus result
        if not os.path.exists(fname):
            raise IOError("Not found: "+fname)
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
            elif act == kREDUCE or act.startswith('ADJOIN('):
                start = nt_idx.pop()
                cat = nt_cat.pop()
                chlds = stack[start:]
                del stack[start:]
                if act != kREDUCE:
                    act_py = str(act)
                    nt_idx.append(start)
                    nt_cat.append(act_py[7:-1])
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

nonpunct_re = regex.compile(u'\p{AlNum}')
s_punct_re = regex.compile(u'[\.\?\!]')

cdef class RNNG_Parser:
    def __init__(self, fname_oracle, fname_model, fname_embeddings,
                 pipelines, use_morph, use_edges, model_settings=None,
                 features=None):
        cdef c_NetworkSettings param
        cdef vector[bool] single
        cdef bool separate_dicts = False
        # if CNN is not initialized yet, do it now
        do_init()
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
        param.compat_version = 1
        if model_settings is not None:
            if 'compat_version' in model_settings:
                param.compat_version = model_settings['compat_version']
            if 'separate_dicts' in model_settings:
                separate_dicts = model_settings['separate_dicts']
        self.use_morph = use_morph
        self.grammar = RNNGrammar(use_morph)
        self.grammar.gram.separate_dicts = separate_dicts
        self.grammar.load_corpus(fname_oracle)
        if fname_embeddings is not None:
            if not os.path.exists(fname_embeddings):
                raise IOError('Not found: '+fname_embeddings)
            ReadEmbeddings_word2vec(
                fname_embeddings,
                &self.grammar.gram.get_pretrain_dict(),
                &self.pretrained)
        self.grammar.Freeze()
        self.model = new c_Model()
        self.parser = new c_ParserBuilder(
            &self.grammar.gram,
            &param, single, self.model,
            &self.pretrained)
        if not os.path.exists(fname_model):
            raise IOError('Not found: '+fname_model)
        load_model(self.model[0], fname_model)
        self.pipelines = pipelines
        if features is None:
            self.features = []
        else:
            self.features = features
        self.parens = ['_LRB_', '_RRB_']
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
        if 'morph' in preproc:
            morph = preproc['morph']
        else:
            morph = ['_'] * len(tags)
        words = line_in.split(' ')
        line_out = line_in
        for pipeline in self.pipelines:
            line_out = pipeline.process_line(line_out)
            line_out = line_out.rstrip('\r\n')
        line_out = line_out.replace('(', self.parens[0])
        line_out = line_out.replace(')', self.parens[1])
        words_unk = line_out.split()
        sent.pos_offset.push_back(0)
        at_start = True
        for i, w in enumerate(words):
            tag = tags[i]
            if self.use_morph:
                morphtag = morph[i]
            else:
                morphtag = "_"
            wu = words_unk[i]
            w_unicode = w.decode('UTF-8')
            w_lower = w_unicode.lower().encode('UTF-8')
            s_w = w
            s_wl = w_lower
            s_wu = wu
            s_tag = tag
            id_w = self.grammar.gram.termdict.Convert(s_w)
            id_wl = self.grammar.gram.get_pretrain_dict().Convert(s_wl)
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
                    sent.pos_val.push_back(id_tag)
            for i, feat in enumerate(self.features):
                if feat is not None:
                    for fval in feat.feature_values(w, at_start):
                        s_tag = '%d%s'%(i+1, fval)
                        id_tag = self.grammar.gram.posdict.Convert(s_tag)
                        sent.pos_val.push_back(id_tag)
            sent.pos_offset.push_back(sent.pos_val.size())
            if at_start:
                if nonpunct_re.search(w):
                    at_start = False
            else:
                if s_punct_re.match(w):
                    at_start = True
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

def need_memory(k):
    global NEEDED_MEM
    if k > NEEDED_MEM:
        NEEDED_MEM = k

def do_init():
    global DONE_INIT
    if DONE_INIT:
        return
    init(['nt-parser', '--cnn-mem', str(NEEDED_MEM)])
    DONE_INIT = True

