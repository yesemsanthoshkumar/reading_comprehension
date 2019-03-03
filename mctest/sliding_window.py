"""
Sliding Window algorithm

https://www.microsoft.com/en-us/research/publication/mctest-challenge-dataset-open-domain-machine-comprehension-text/
"""

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

OPTIONS = ['A', 'B', 'C', 'D']
QUESTIONS = [1, 2, 3, 4]
OPTIONS = [1, 2, 3, 4]
QUESTION_COLUMNS = [
    'id', 'auth_wrktime', 'passage',
    'q1', 'q11', 'q12', 'q13', 'q14',
    'q2', 'q21', 'q22', 'q23', 'q24',
    'q3', 'q31', 'q32', 'q33', 'q34',
    'q4', 'q41', 'q42', 'q43', 'q44'
]
ANSWER_COLUMNS = ['q1', 'q2', 'q3', 'q4']
STOPWORDS = []

with open('mctest/data/stopwords.txt', 'r') as infl:
    STOPWORDS = infl.read().split('\n')


class SlidingWindow(object):
    __options = ['A', 'B', 'C', 'D']
    stopwords = set(stopwords.words('english'))
    # stopwords = STOPWORDS

    def __init__(self, question_df, answer_df):
        self.question_df = question_df
        self.answer_df = answer_df

    def question_preprocess(self, question):
        prcd_question = []
        for qw in question.replace('?', '').split():
#             if qw not in self.stopwords:
            prcd_question.append(qw)
        return ' '.join(prcd_question)

    def preprocess(self):
        self.question_df['passage'] = self.question_df['passage'].map(lambda x: x.replace('\\newline', ''))

        self.word_counts = defaultdict(lambda: 0)
        for passage in self.question_df['passage']:
            for token in passage.split():
                self.word_counts[token.lower()] += 1

        self.inv_counts = {k: np.log(1 + (1/v)) for k, v in self.word_counts.items()}

        for i in range(1, 5):
            q_type = 'q{0}_type'.format(i)
            q = 'q{0}'.format(i)
            self.question_df[q_type] = self.question_df[q].map(lambda x: x.split(':')[0])
            self.question_df[q] = self.question_df[q].map(lambda x: self.question_preprocess(x.split(':')[1]))

    def predict(self, with_dist=False):
        ans_calc = []
        for rw in self.question_df.iterrows():
            ans_row = []
            for qi in [1, 2, 3, 4]:
                ans = []
                for ai in [1, 2, 3, 4]:
                    score = self.sliding_window_score(rw[1], qi=qi, ai=ai)
                    if with_dist:
                        dist = distance_based(rw[1], qi=qi, ai=a1)
                        ans.append(score - dist)
                    else:
                        ans.append(score)
#                Return argmax i sw 1..4
                calc_option = self.__options[np.argmax(ans)]
                ans_row.append(calc_option)
            ans_calc.append(ans_row)

        return pd.DataFrame(ans_calc, columns=['a1', 'a2', 'a3', 'a4'])

    def score(self, ans_df):
        eval_df = pd.merge(
            self.question_df[['q1_type', 'q2_type', 'q3_type', 'q4_type']],
            pd.merge(
                self.answer_df,
                ans_df,
                how='inner',
                left_index=True,
                right_index=True
            ),
            how='inner',
            left_index=True,
            right_index=True
        )

        # Compute scores for questions with single answer and multiple answers separately
        score_single = 0
        score_multiple = 0
        question_single = 0
        question_multiple = 0

        for i in range(1, 5):
            question_single += eval_df[eval_df['q%s_type' % i] == 'one'].shape[0]
            question_multiple += eval_df[eval_df['q%s_type' % i] == 'multiple'].shape[0]

            score_single += (np.logical_and(
                eval_df['q%s_type' % i] == 'one',
                eval_df['q%s' % i] == eval_df['a%s' % i])
            ).sum()

            score_multiple += (np.logical_and(
                eval_df['q%s_type' % i] == 'multiple',
                eval_df['q%s' % i] == eval_df['a%s' % i])
            ).sum()

        return {
            'One': {
                'Total': question_single,
                'Correct': score_single,
                'score': score_single / question_single
            },
            'Multiple': {
                'Total': question_multiple,
                'Correct': score_multiple,
                'score': score_multiple / question_multiple
            }
        }

    def sliding_window_score(self, story, window=3, qi=1, ai=1):
        tokens = story['passage'].lower().split()
        question_tokens = story['q%s' % qi].lower().split()
        answer_tokens = story['q%s%s' % (qi, ai)].lower().split()

        max_overlap_score = 0
#         S = A U Q
        target_set = set(question_tokens + answer_tokens)

        for pi, _ in enumerate(tokens):
            overlap_score = 0

            try:
                for w in range(window):
#                     IC(P j+w) if P j+w belongs to S
                    if tokens[pi + w] in target_set and tokens[pi + w] not in self.stopwords:
                        overlap_score += self.inv_counts[tokens[pi + w]]
            except IndexError:
                break

    #             print("Overlap score: ", overlap_score)
    #             print("Max Overlap score: ", max_overlap_score)
            if overlap_score > max_overlap_score:
                tokens_overlapped = tokens[pi : pi + window]
                print("Max=%f\tScore=%f\tOverlapped=%s\tPassage=%s" % (
                    max_overlap_score, overlap_score, ' '.join(tokens_overlapped), ' '.join(target_set)
                ))
                max_overlap_score = overlap_score
    #     print("Ans: ", ai, " Max Overlap Score: ", max_overlap_score)
        return max_overlap_score


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Sliding window algorithm")
    parser.add_argument(
        '--with-distance',
        # TODO: Find the bug when the distance param is used
        dest='uses_distance',
        action='store_true',
        help='Whether sliding window should use distance metric'
    )

    arguments = parser.parse_args()

    # Dev set
    question_dev_160 = pd.read_csv("mctest/data/MCTest/mc160.dev.tsv", delimiter='\t',
            names=QUESTION_COLUMNS
    )
    answer_dev_160 = pd.read_csv(
        "mctest/data/MCTest/mc160.dev.ans",
        delimiter='\t',
        names=ANSWER_COLUMNS
    )

    # Train set
    question_train_160 = pd.read_csv("mctest/data/MCTest/mc160.train.tsv", delimiter='\t',
        names=QUESTION_COLUMNS
    )
    answer_train_160 = pd.read_csv(
        "mctest/data/MCTest/mc160.train.ans",
        delimiter='\t',
        names=ANSWER_COLUMNS
    )

    ques_df = pd.concat([question_dev_160, question_train_160])
    ans_df = pd.concat([answer_dev_160, answer_train_160])
    sw_dev_train = SlidingWindow(ques_df, ans_df)

    sw_dev_train.preprocess()

    dev_train_predictions = sw_dev_train.predict(with_dist=arguments.uses_distance)

    score = sw_dev_train.score(dev_train_predictions)

    print(score)
