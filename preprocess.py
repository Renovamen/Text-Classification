import datasets.preprocess as preprocess
from utils import opts

if __name__ == '__main__':

    config = opts.parse_opt()
    preprocess.run_prepro(
        csv_folder = config.dataset_path,
        output_folder = config.output_path,
        sentence_limit = config.sentence_limit,
        word_limit = config.word_limit,
        min_word_count = config.min_word_count
    )