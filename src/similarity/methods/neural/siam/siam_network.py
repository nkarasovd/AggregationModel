import pickle
from typing import List, Tuple, Iterable

import torch

from src.similarity.methods.neural.neural_base import NeuralModel
from src.similarity.methods.neural.siam.classifier import StackClassifier


class SiamMultiModalModel(NeuralModel):
    def __init__(self, encoders, agg, **kwargs):
        super(SiamMultiModalModel, self).__init__()
        self.agg = agg(encoders)
        self.classifier = StackClassifier(input_dim=self.agg.out_dim(), **kwargs)
        self.cache = {}

    def fit(self, sim_train_data: List[Tuple[int, int, int]] = None, unsup_data: Iterable[int] = None):
        pass

    def get_agg(self, stack_id):
        # return self.agg(stack_id)
        if self.training:
            self.cache = {}
            return self.agg(stack_id)
        else:
            if stack_id not in self.cache:
                self.cache[stack_id] = self.agg(stack_id)
            return self.cache[stack_id]

    def forward(self, stack_id1, stack_id2):
        return self.classifier(self.get_agg(stack_id1), self.get_agg(stack_id2))

    def predict(self, anchor_id, stack_ids):
        with torch.no_grad():
            y_pr = []
            for stack_id in stack_ids:
                y_pr.append(self.forward(anchor_id, stack_id).cpu().numpy()[1])
            return y_pr

    def name(self):
        return self.agg.name() + "_siam_" + self.classifier.name()

    def train(self, mode=True):
        super().train(mode)

    def opt_params(self):
        return self.agg.opt_params() + self.classifier.opt_params()

    def save_modules(self, encoder_path: str, classifier_path: str):
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.agg, f)

        with open(classifier_path, 'wb') as f:
            pickle.dump(self.classifier, f)

    def save(self, model_path: str):
        torch.save(self, model_path)

    @staticmethod
    def load(model_path: str):
        model = torch.load(model_path)
        model.eval()
        return model
