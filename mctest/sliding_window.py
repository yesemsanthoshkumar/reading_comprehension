"""
Sliding Window algorithm

https://www.microsoft.com/en-us/research/publication/mctest-challenge-dataset-open-domain-machine-comprehension-text/
"""

import argparse
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

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
puncts = r',|\.|\'(?:s)'

with open('mctest/data/stopwords.txt', 'r') as infl:
    STOPWORDS = infl.read().split('\n')


class SlidingWindow(object):
    __options = ['A', 'B', 'C', 'D']
    stopwords = set(stopwords.words('english'))
    stopwords = stopwords.union(set(STOPWORDS))

    def __init__(self, question_df, answer_df, window):
        self.question_df = question_df
        self.answer_df = answer_df
        self.window_size = window

    @staticmethod
    def remove_puncts(word):
        return re.sub(puncts, '', word)

    def question_preprocess(self, question):
        prcd_question = []
        for qw in question.replace('?', '').split():
            wrd = SlidingWindow.remove_puncts(qw)
            if len(wrd) > 2 and wrd not in self.stopwords:
                prcd_question.append(wrd)
        return ' '.join(prcd_question)

    def preprocess(self):
        self.question_df['passage'] = self.question_df['passage'].map(lambda x: x.replace('\\newline', ''))

        self.word_counts = defaultdict(lambda: 0)
        for passage in self.question_df['passage']:
            for token in passage.split():
                self.word_counts[SlidingWindow.remove_puncts(token.lower())] += 1

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
                        story = rw[1]
                        dist = self.calculate_min_distance(
                            passage=story['passage'],
                            question=story['q%s' % qi],
                            answer=story['q%s%s' % (qi, ai)]
                        )
                        ans.append(score - dist)
                    else:
                        ans.append(score)
            #    Return argmax i sw 1..4
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
            'one': {
                'total': question_single,
                'correct': score_single,
                'score': score_single / question_single
            },
            'multiple': {
                'total': question_multiple,
                'correct': score_multiple,
                'score': score_multiple / question_multiple
            }
        }

    def sliding_window_score(self, story, qi=1, ai=1):
        tokens = [
            SlidingWindow.remove_puncts(x)
            for x in story['passage'].lower().split()
            if SlidingWindow.remove_puncts(x) not in self.stopwords
        ]
        question_tokens = story['q%s' % qi].lower().split()
        answer_tokens = story['q%s%s' % (qi, ai)].lower().split()

        max_overlap_score = 0
        # S = A U Q
        target_set = set(question_tokens + answer_tokens)

        for pi, _ in enumerate(tokens):
            overlap_score = 0

            try:
                for w in range(self.window_size):
                    # IC(P j+w) if P j+w belongs to S
                    if tokens[pi + w] in target_set and tokens[pi + w] not in self.stopwords:
                        overlap_score += self.inv_counts[tokens[pi + w]]
            except IndexError:
                break

                # print("Overlap score: ", overlap_score)
                # print("Max Overlap score: ", max_overlap_score)
            if overlap_score > max_overlap_score:
                tokens_overlapped = tokens[pi : pi + self.window_size]
                # print("Max=%f\tScore=%f\tOverlappedPassageTokens=%s\tQuestion=%s" % (
                #     max_overlap_score, overlap_score, ' '.join(tokens_overlapped), ' '.join(question_tokens)
                # ))
                max_overlap_score = overlap_score
        # print("Ans: ", ai, " Max Overlap Score: ", max_overlap_score)
        return max_overlap_score

    def calculate_min_distance(self, passage, question, answer):
        ques_tokens = []
        ans_tokens = []
        passage_tokens = word_tokenize(passage.lower())

        for q in word_tokenize(question.lower()):
            if q in passage_tokens and q not in self.stopwords:
                ques_tokens.append(q)
        for a in word_tokenize(answer.lower()):
            if a in passage_tokens and a not in self.stopwords:
                ans_tokens.append(a)

        if len(ques_tokens) == 0 or len(ans_tokens) == 0:
            return 1

        min_dist = None
        for  qt in ques_tokens:
            qt_pos = [i for i, x in enumerate(passage_tokens) if x == qt]
            for atok in ans_tokens:
                atok_pos = [i for i, x in enumerate(passage_tokens) if x == atok]
                for qps in qt_pos:
                    for aps in atok_pos:
                        dist = abs(qps - aps)
                        # print(qt, qt_pos)
                        # print(atok, atok_pos)
                        # print(dist, qps, aps)
                        if min_dist is None or dist < min_dist:
                            min_dist = dist
                            # print("Min dist changed")

        return min_dist

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Sliding window algorithm")
    parser.add_argument(
        '--with-distance',
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
    sw_dev_train = SlidingWindow(ques_df, ans_df, window=3)

    sw_dev_train.preprocess()

    dev_train_predictions = sw_dev_train.predict(with_dist=arguments.uses_distance)

    score_160 = sw_dev_train.score(dev_train_predictions)

    question_dev_500 = pd.read_csv("mctest/data/MCTest/mc500.dev.tsv", delimiter='\t',
            names=QUESTION_COLUMNS
    )
    answer_dev_500 = pd.read_csv(
        "mctest/data/MCTest/mc500.dev.ans",
        delimiter='\t',
        names=ANSWER_COLUMNS
    )
    question_train_500 = pd.read_csv("mctest/data/MCTest/mc500.train.tsv", delimiter='\t',
        names=QUESTION_COLUMNS
    )
    answer_train_500 = pd.read_csv(
        "mctest/data/MCTest/mc500.train.ans",
        delimiter='\t',
        names=ANSWER_COLUMNS
    )

    ques_df = pd.concat([question_dev_500, question_train_500])
    ans_df = pd.concat([answer_dev_500, answer_train_500])
    sw_dev_train = SlidingWindow(ques_df, ans_df, window=3)

    sw_dev_train.preprocess()

    dev_train_predictions = sw_dev_train.predict(with_dist=arguments.uses_distance)

    score = sw_dev_train.score(dev_train_predictions)

    print("160")
    print("one:\n\tTotal:{0} correct:{1} score:{2}".format(score_160['one']['total'], score_160['one']['correct'], score_160['one']['score']))
    print("multiple:\n\tTotal:{0} correct:{1} score:{2}".format(score_160['multiple']['total'], score_160['multiple']['correct'], score_160['multiple']['score']))

    print("500")
    print("one:\n\tTotal:{0} correct:{1} score:{2}".format(score['one']['total'], score['one']['correct'], score['one']['score']))
    print("multiple:\n\tTotal:{0} correct:{1} score:{2}".format(score['multiple']['total'], score['multiple']['correct'], score['multiple']['score']))
