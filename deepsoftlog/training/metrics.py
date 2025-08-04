import numpy as np
from sklearn.metrics import average_precision_score

from ..data import Query

def debug_expr_tree(expr, depth=0):
    print(f"{'  ' * depth}Examining node: {expr}")
    print(f"{'  ' * depth}Type: {type(expr)}")
    print(f"{'  ' * depth}Dir: {dir(expr)}")
    if hasattr(expr, 'arguments'):
        print(f"{'  ' * depth}Arguments:")
        for i, arg in enumerate(expr.arguments):
            print(f"{'  ' * depth}  Argument {i}:")
            debug_expr_tree(arg, depth + 1)

def get_metrics(query: Query, results, dataset) -> dict[str, float]:
    # print("\n=== Starting Expression Tree Debug ===")
    # debug_expr_tree(query.query)
    # print("=== End Expression Tree Debug ===\n")
    # print("get_metrics")
    # print(query)
    # print(type(query))
    # print(query.query)
    # print(type(query.query))
    metrics = boolean_metrics(results, query)
    # print(type(query.query))
    if not query.query.is_ground():
        assert query.p == 1
        metrics.update(rank_metrics(results, dataset))
    return metrics


def boolean_metrics(results, query) -> dict[str, float]:
    pred = max(results.values(), default=0)
    print(f"Query: {query}")
    print(f"Results: {results}")
    print(f"Pred: {pred}")
    if not results:
        print("No results found.")
    diff = abs(query.p - pred)
    print(f"Diff: {diff}")
    return {
        "diff": diff,
        "target": query.p,
        "pred": pred,
        "threshold_accuracy": 1 if diff <= 0.5 else 0,
    }


def rank_metrics(results, dataset) -> dict[str, float]:
    results = sorted(results.items(), key=lambda x: -x[1])
    for i, (result, _) in enumerate(results):
        if result in dataset:
            rank = i + 1
            return {
                "mrr": 1 / rank,
                "hits@1": 1 if rank <= 1 else 0,
                "hits@3": 1 if rank <= 3 else 0,
                "hits@10": 1 if rank <= 10 else 0,
            }

    return {"mrr": 0, "hits@1": 0, "hits@3": 0, "hits@10": 0}


def aggregate_metrics(metrics_list: list) -> dict:
    result = {}
    metric_names = metrics_list[0].keys()
    for metric_name in metric_names:
        result[metric_name] = np.nanmean([x[metric_name] for x in metrics_list])

    targets = np.array([x['target'] for x in metrics_list])
    preds = np.array([x['pred'] for x in metrics_list])
    result['auc'] = average_precision_score(targets, preds)
    return result
