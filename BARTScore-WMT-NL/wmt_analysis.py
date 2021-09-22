# %%
from utils import *
from copy import deepcopy
from tqdm import trange
from tqdm import tqdm
import os 


from analysis import WMTStat

def truncate_print(l, n=10):
    """ Print the first n items of a list"""
    for i, x in enumerate(l):
        if i == n:
            print('...')
            break
        print(x)


def main():
    wmt_stat = WMTStat('WMT/kk-en/scores.pkl')
    wmt_stat.print_ref_len()
    print('All metrics')
    print(wmt_stat.metrics) # Print out all metrics
    print('\n')
    print('All k-tau')
    wmt_stat.print_ktau()
    print('\n')

main()


