from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.similarity.methods.base import SimStackModel
from src.similarity.methods.classic.tfidf import IntTfIdfComputer
from src.similarity.preprocess.seq_coder import SeqCoder


class DurfexModel(SimStackModel):
    def __init__(self, coder: SeqCoder, ns: Tuple[int, ...] = None):
        self.tfidf = IntTfIdfComputer(coder, ns)
        self.coder = coder

    def fit(self, sim_train_data: List[Tuple[int, int, int]] = None, unsup_data: List[int] = None) -> 'DurfexModel':
        self.tfidf.fit(unsup_data)
        return self

    def predict(self, anchor_id: int, stack_ids: List[int]) -> List[float]:
        scores = []

        def to_dict(stack_id):
            return {word: tf * np.exp(idf - 1) for word, (tf, idf) in self.tfidf.transform(stack_id).items()}

        anchor = to_dict(anchor_id)
        for stack_id in stack_ids:
            array = DictVectorizer().fit_transform([anchor, to_dict(stack_id)])
            scores.append(cosine_similarity(array[0:], array[1:])[0, 0])

        return scores

    def name(self) -> str:
        return self.coder.name() + "_durfex"
