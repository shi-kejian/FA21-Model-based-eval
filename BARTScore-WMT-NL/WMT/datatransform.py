import argparse
import os
import time
import numpy as np
from utils import *
from tqdm import tqdm
import itertools
import re

from transformations.butter_fingers_perturbation.transformation import ButterFingersPerturbation
from transformations.gender_swap.transformation import GenderSwap
from transformations.synonym_substitution.transformation import SynonymSubstitution

REF_HYPO = read_file_to_list('files/tiny_ref_hypo_prompt.txt')


def main():

        kk-en-syno-both/data.pkl
        device='cuda:0'
        device = device
        data = read_pickle(file_path)

        # ------------
        data = dict(itertools.islice(data.items(),3000))
        print("len of pkl:", len(data))
        # ------------ *
        
        for doc_id in data:
            ref = data[doc_id]['ref']
            cand = data[doc_id]['better']['sys']

            ## ----synonym sub -both -----
            syno_transformer = SynonymSubstitution(max_outputs=1)
            new_ref = syno_transformer.generate(ref)[0]
            new_cand = syno_transformer.generate(cand)[0]

            data[doc_id]['ref'] = new_ref
            data[doc_id]['better']['sys'] = new_cand
            ## ------------------

            ## ----synonym sub -only candi-----
            # syno_transformer = SynonymSubstitution(max_outputs=1)
            # new_ref = ref
            # new_cand = syno_transformer.generate(cand)[0]
            ## ------------------
        print(len(data))

        save_pickle(data,"./kk-en-syno-both/data_preped.pkl")

main()
