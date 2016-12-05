'''
replaces the POS tags and tokens in RNNG's output while
keeping the generated edge labels
'''
import re
import sys

re_preterm = re.compile(r'\(([^\s\):]+)(?::(\S+))? ([^\s\)]+)\)')

def get_tags_edges_tokens(line):
    line_strip = line.rstrip()
    tags = []
    edges = []
    tokens = []
    for m in re_preterm.finditer(line_strip):
        edge_label = m.group(2)
        if edge_label is None:
            edge_label = '--'
        tags.append(m.group(1))
        edges.append(edge_label)
        tokens.append(m.group(3))
    return tags, edges, tokens

def main():
    if len(sys.argv) != 3:
        raise NotImplementedError('Program only takes two arguments: the gold dev set and the output file dev set')
    gold_file = open(sys.argv[1], 'r')
    sys_file = open(sys.argv[2], 'r')
    gold_lines = gold_file.readlines()
    sys_lines = sys_file.readlines()
    gold_file.close()
    sys_file.close()
    assert len(gold_lines) == len(sys_lines)
    for gold_line, sys_line in zip(gold_lines, sys_lines):
        sys_tags, sys_edges, sys_tokens = get_tags_edges_tokens(sys_line)
        gold_tags, gold_edges, gold_tokens = get_tags_edges_tokens(gold_line)
        assert len(sys_tokens) == len(gold_tokens), (sys_tokens, gold_tokens)
        output = []
        gold_idx = 0
        sys_line = sys_line.rstrip()
        for i, match in enumerate(re_preterm.finditer(sys_line)):
            output.append(sys_line[gold_idx:match.start()])
            output.append('(%s:%s %s)'%(
                gold_tags[i], sys_edges[i], gold_tokens[i]))
            assert sys_tags[i] == 'XX', sys_tags[i]
            assert gold_tags[i] != 'XX', gold_tags[i]
            gold_idx = match.end()
        output.append(sys_line[gold_idx:])
        print ''.join(output)

if __name__ == '__main__':
    main()
