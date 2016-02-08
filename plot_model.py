import numpy as np
import matplotlib.pyplot as plot
import cPickle
import ribohmm
import utils
import pdb

transitions = []
emissions = []

colors = ['r','b','#888888']
bs = [1,10,100,1000,10000,100000]
offsets = np.array([-1,-0.65,-0.3,0.05,0.4,0.75])
xvals = np.array([1,4,7,10,13,16,19,22,25])
for b in bs:
    handle = open("test/hg19_test_%d.pkl"%b,'r')
    transitions.append(cPickle.load(handle))
    emissions.append(cPickle.load(handle))
    handle.close()

states = ['5UTS','5UTS+','TIS','TIS+','TES','TTS-','TTS','3UTS-','3UTS']

for r,rl in enumerate(utils.READ_LENGTHS):

    # plot periodicity comparisons
    figure = plot.figure()
    subplot = figure.add_axes([0.1,0.1,0.8,0.8])
    for j,b in enumerate(bs):
        period = emissions[j].periodicity[r]
        for k in xrange(3):
            height = period[:,k]
            bottom = period[:,:k+1].sum(1)-height
            subplot.bar(xvals+offsets[j], height, bottom=bottom, linewidth=0, width=0.25, color=colors[k])
    figure.savefig("test/hg19_test_period_%dbp.pdf"%rl, dpi=450)

    # plot alpha comparisons
    figure = plot.figure()
    subplot = figure.add_axes([0.1,0.1,0.8,0.8])
    for j,b in enumerate(bs):
        height = emissions[j].rate_alpha[r]
        subplot.bar(xvals+offsets[j], height, linewidth=0, width=0.25, color=colors[k])
    figure.savefig("test/hg19_test_alpha_%dbp.pdf"%rl, dpi=450)

    # plot beta comparisons
    figure = plot.figure()
    subplot = figure.add_axes([0.1,0.1,0.8,0.8])
    for j,b in enumerate(bs):
        height = emissions[j].rate_beta[r]
        subplot.bar(xvals+offsets[j], height, linewidth=0, width=0.25, color=colors[k])
    figure.savefig("test/hg19_test_beta_%dbp.pdf"%rl, dpi=450)
