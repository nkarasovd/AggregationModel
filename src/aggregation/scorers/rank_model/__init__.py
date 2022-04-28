from typing import Any, Dict

from src.aggregation.scorers.rank_model.linear_rank_model import LinearRankModel
from src.aggregation.scorers.rank_model.rank_model import RankModel


def get_rank_model(model_config: Dict[str, Any]) -> RankModel:
    if model_config['model_type'] == 'linear_model':
        return LinearRankModel.load_from_config(model_config)
    else:
        raise ValueError("Model type is not match")
