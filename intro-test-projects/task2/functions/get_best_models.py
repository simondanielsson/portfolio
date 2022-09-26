from typing import List, Tuple

def get_best_models(scores: dict, subtask: int) -> List[Tuple[str, object]]:
    """
    Gets a dictorionary of models and their scores and outputs the best models per label column.
    For subtask 1 and 2 the best is the one with the largest average roc auc across all labels
    Also prints its average roc auc.

    scores = {"<model_name>": {
            "<label_name1>": (trained_model_object, (train_score1, val_score1)),
            "<label_name2>": (trained_model_object2, (train_score2, val_score2)),
        },
        ...
    }

    return: [(label_name, trained_model_object), ...]
    """
