import sys
import fileinput
import re
best = 0
best_epoch = 0
avg = 0
fscore = 0
epoch = 0
report_re = re.compile(r'dev .* epoch=([0-9]+\.[0-9]+)\).* f1: ([0-9]+(?:\.[0-9]+)?)')
update_re = re.compile(r'\*\*SHUFFLE|\[epoch=[0-9]+ eta=')
other_output = []
for l in fileinput.input():
    m = report_re.search(l)
    if m:
        fscore = float(m.group(2))
        epoch = float(m.group(1))
        avg = 0.9 * avg + 0.1 * fscore
        if (fscore > best):
            best = fscore
            best_epoch = epoch
            other_output = []
    else:
        m = update_re.match(l)
        if not m:
            other_output.append(l)
for l in other_output[-5:]:
    print l,
print "f1: %5.2f  epoch: %5.2f  (avg %5.2f best: %5.2f @ epoch %5.2f"%(
    fscore, epoch, avg, best, best_epoch)
