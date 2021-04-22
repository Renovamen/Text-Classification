from typing import Tuple, Dict

from . import ag_news, dbpedia, yelp_polarity, yelp_full, yahoo_answers, \
    amazon_polarity, amazon_full

def get_label_map(dataset: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    if dataset == 'ag_news':
        classes = ag_news.classes
    elif dataset == 'dbpedia':
        classes = dbpedia.classes
    elif dataset == 'yelp_polarity':
        classes = yelp_polarity.classes
    elif dataset == 'yelp_full':
        classes = yelp_full.classes
    elif dataset == 'yahoo_answers':
        classes = yahoo_answers.classes
    elif dataset == 'amazon_polarity':
        classes = amazon_polarity.classes
    elif dataset == 'amazon_full':
        classes = amazon_full.classes
    else:
        raise Exception("Dataset not supported: ", dataset)

    label_map = {k: v for v, k in enumerate(classes)}
    rev_label_map = {v: k for k, v in label_map.items()}

    return label_map, rev_label_map
