from datasets import run_doc_prepro, run_sent_prepro
from utils import parse_opt

if __name__ == '__main__':
    config = parse_opt()

    if config.model_name in ['han']:
        run_doc_prepro(
            csv_folder = config.dataset_path,
            output_folder = config.output_path,
            word_limit = config.word_limit,
            sentence_limit = config.sentence_limit,
            min_word_count = config.min_word_count
        )
    else:
        run_sent_prepro(
            csv_folder = config.dataset_path,
            output_folder = config.output_path,
            word_limit = config.word_limit,
            min_word_count = config.min_word_count
        )
