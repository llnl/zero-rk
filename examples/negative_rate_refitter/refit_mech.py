
import os

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
from zerork.negative_rate_refitter import negative_rate_refitter

def save_multipage(filename):
    pp = PdfPages(filename)
    figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()

if __name__ == '__main__':
    nrr = negative_rate_refitter('negative_rates.mech')
    nrr.write_mech('refit.mech')
    save_multipage('rate_comparison.pdf')


