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

def sent_contains_HE(text):
    exp = re.compile(r'he',re.IGNORECASE)
    if exp.search(text):
        print("YESSS")
    return exp.search(text)

def gender_swap(text):

    name = ""
    return name


def syno_subst(ref, cand):
    pass



class Scorer:
    """ Support BLEU, CHRF, BLEURT, PRISM, COMET, BERTScore, BARTScore """

    def __init__(self, file_path, device='cuda:0'):
        """ file_path: path to the pickle file
            All the data are normal capitalized, not tokenied, including src, ref, sys
        """
        self.device = device
        self.data = read_pickle(file_path)

        # ------------
        self.data = dict(itertools.islice(self.data.items(),3000))
        print("len of pkl:", len(self.data))
        # ------------ *
        
        print(f'Data loaded from {file_path}.')
        self.refs, self.betters, self.worses = [], [], []
        for doc_id in self.data:
            ref = self.data[doc_id]['ref']
            cand = self.data[doc_id]['better']['sys']

            # ----original data----
            # self.refs.append(ref)
            # self.betters.append(cand)
            # ------------------

            ## ----gender swap-----
            # gender_transformer = GenderSwap(max_outputs=1)
            # new_ref = gender_transformer.generate(ref)[0]
            # new_cand = gender_transformer.generate(cand)[0]
            ## ------------------

            ## ----synonym sub -both -----
            # syno_transformer = SynonymSubstitution(max_outputs=1)
            # new_ref = syno_transformer.generate(ref)[0]
            # new_cand = syno_transformer.generate(cand)[0]
            ## ------------------

            ## ----synonym sub -only candi-----
            # syno_transformer = SynonymSubstitution(max_outputs=1)
            # new_ref = ref
            # new_cand = syno_transformer.generate(cand)[0]
            ## ------------------

            butter_transformer = ButterFingersPerturbation(max_outputs=1)
            new_ref = ref
            new_cand = butter_transformer.generate(cand)[0]

            # print(ref)
            # print(new_ref)
            self.refs.append(new_ref)
            self.betters.append(new_cand)
            

            self.worses.append(self.data[doc_id]['worse']['sys'])

        print(len(self.data))
       
    def save_data(self, path):
        save_pickle(self.data, path)


    def record(self, scores_better, scores_worse, name):
        """ Record the scores from a metric """
        for doc_id in self.data:
            self.data[doc_id]['better']['scores'][name] = str(scores_better[doc_id])
            self.data[doc_id]['worse']['scores'][name] = str(scores_worse[doc_id])

    def score(self, metrics):
        for metric_name in metrics:
            if metric_name == 'bleu':
                from sacrebleu import corpus_bleu
                from sacremoses import MosesTokenizer

                def run_sentence_bleu(candidates: list, references: list) -> list:
                    """ Runs sentence BLEU from Sacrebleu. """
                    tokenizer = MosesTokenizer(lang='en')
                    candidates = [tokenizer.tokenize(mt, return_str=True) for mt in candidates]
                    references = [tokenizer.tokenize(ref, return_str=True) for ref in references]
                    assert len(candidates) == len(references)
                    bleu_scores = []
                    for i in range(len(candidates)):
                        bleu_scores.append(corpus_bleu([candidates[i], ], [[references[i], ]]).score)
                    return bleu_scores

                start = time.time()
                print(f'Begin calculating BLEU.')
                scores_better = run_sentence_bleu(self.betters, self.refs)
                scores_worse = run_sentence_bleu(self.worses, self.refs)
                print(f'Finished calculating BLEU, time passed {time.time() - start}s.')
                self.record(scores_better, scores_worse, 'bleu')

            elif metric_name == 'bleurt':
                from bleurt import score

                def run_bleurt(
                        candidates: list, references: list, checkpoint: str = "models/bleurt-large-512"
                ):
                    scorer = score.BleurtScorer(checkpoint)
                    scores = scorer.score(references=references, candidates=candidates)
                    return scores

                start = time.time()
                print(f'Begin calculating BLEURT.')
                scores_better = run_bleurt(self.betters, self.refs)
                scores_worse = run_bleurt(self.worses, self.refs)
                print(f'Finished calculating BLEURT, time passed {time.time() - start}s.')
                self.record(scores_better, scores_worse, 'bleurt')

            elif metric_name == 'bert_score':
                import bert_score

                def run_bertscore(mt: list, ref: list):
                    """ Runs BERTScores and returns precision, recall and F1 BERTScores ."""
                    _, _, f1 = bert_score.score(
                        cands=mt,
                        refs=ref,
                        idf=False,
                        batch_size=32,
                        lang='en',
                        rescale_with_baseline=False,
                        verbose=True,
                        nthreads=4,
                    )
                    return f1.numpy()

                start = time.time()
                print(f'Begin calculating BERTScore.')
                scores_better = run_bertscore(self.betters, self.refs)
                scores_worse = run_bertscore(self.worses, self.refs)
                print(f'Finished calculating BERTScore, time passed {time.time() - start}s.')
                self.record(scores_better, scores_worse, 'bert_score')

            elif metric_name == 'bart_score' or metric_name == 'bart_score_cnn' or metric_name == 'bart_score_para':
                from bart_score import BARTScorer

                def run_bartscore(scorer, mt: list, ref: list):
                    hypo_ref = np.array(scorer.score(mt, ref, batch_size=4))
                    ref_hypo = np.array(scorer.score(ref, mt, batch_size=4))
                    avg_f = 0.5 * (ref_hypo + hypo_ref)
                    return avg_f

                # Set up BARTScore
                if 'cnn' in metric_name:
                    bart_scorer = BARTScorer(device=self.device, checkpoint='facebook/bart-large-cnn')
                elif 'para' in metric_name:
                    bart_scorer = BARTScorer(device=self.device, checkpoint='facebook/bart-large-cnn')
                    bart_scorer.load()
                else:
                    bart_scorer = BARTScorer(device=self.device, checkpoint='facebook/bart-large')

                start = time.time()
                print(f'Begin calculating BARTScore.')
                scores_better = run_bartscore(bart_scorer, self.betters, self.refs)
                scores_worse = run_bartscore(bart_scorer, self.worses, self.refs)
                print(f'Finished calculating BARTScore, time passed {time.time() - start}s.')
                self.record(scores_better, scores_worse, metric_name)

    def report_avg_score(self,metrics):
        import statistics
        for metric_name in metrics: 
            scores = []
            for doc_id in self.data:
                scores.append(float(self.data[doc_id]['better']['scores'][metric_name]))
            print(f'For {metric_name}, average score over {len(self.data)} entries is:', statistics.mean(scores))



def main():
    parser = argparse.ArgumentParser(description='Scorer parameters')
    parser.add_argument('--file', type=str, required=True,
                        help='The data to load from.')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='The device to run on.')
    parser.add_argument('--gender_swap', type=bool, required=False,
                        help='whether do gender_swap')
    parser.add_argument('--syno_sub', type=bool, required=False,
                        help='whether do synonym substitution')
    parser.add_argument('--output', type=str, required=True,
                        help='The output path to save the calculated scores.')
    parser.add_argument('--bleu', action='store_true', default=False,
                        help='Whether to calculate BLEU')
    parser.add_argument('--chrf', action='store_true', default=False,
                        help='Whether to calculate CHRF')
    parser.add_argument('--bleurt', action='store_true', default=False,
                        help='Whether to calculate BLEURT')
    parser.add_argument('--bert_score', action='store_true', default=False,
                        help='Whether to calculate BERTScore')
    parser.add_argument('--bart_score', action='store_true', default=False,
                        help='Whether to calculate BARTScore')
    args = parser.parse_args()

    scorer = Scorer(args.file, args.device)

    METRICS = []
    TRANSFORMATION = []
    if args.bleu:
        METRICS.append('bleu')
    if args.bleurt:
        METRICS.append('bleurt')
    if args.bert_score:
        METRICS.append('bert_score')
    if args.bart_score:
        METRICS.append('bart_score')
    if args.gender_swap:
        TRANSFORMATION.append('gender')
    if args.syno_sub:
        TRANSFORMATION.append('synonym')

    scorer.score(METRICS)
    scorer.report_avg_score(METRICS)
    scorer.save_data(args.output)


if __name__ == '__main__':
    main()

"""
python score.py --file kk-en/data.pkl --device cuda:0 --output kk-en/scores.pkl --bleu --chrf --bleurt --prism --comet --bert_score --bart_score --bart_score_cnn --bart_score_para

python score.py --file lt-en/scores.pkl --device cuda:3 --output lt-en/scores.pkl --bart_score --bart_score_cnn --bart_score_para
"""
