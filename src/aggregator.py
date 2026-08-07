import torch
import copy
from models import BaseModel

class Aggregator():
    def __init__(self) -> None:
        pass

    def fed_avg(self, models: dict[str, BaseModel]) -> dict[str, torch.Tensor]:
        first_key = next(iter(models))
        avg_model = copy.deepcopy(models[first_key])
        avg_weights = avg_model.state_dict()

        for key in avg_weights.keys():
            for model_key in models.keys():
                # Skip first model (weights are already copied)
                if model_key == first_key:
                    continue
                model = models[model_key].state_dict()
                avg_weights[key] += model[key]
            avg_weights[key] = torch.div(avg_weights[key], len(models))
        return avg_weights

    def update_model(self, model: BaseModel, new_weights: torch.Tensor) -> BaseModel:
        return model.load_state_dict(new_weights)