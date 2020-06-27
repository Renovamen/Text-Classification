from datasets.preprocess import sentence, document
from utils import opts

if __name__ == '__main__':

    config = opts.parse_opt()

    if config.model_name in ['han']:
        document.run_prepro(
            csv_folder = config.dataset_path,
            output_folder = config.output_path,
            word_limit = config.word_limit,
            sentence_limit = config.sentence_limit,
            min_word_count = config.min_word_count
        )
    else:
        sentence.run_prepro(
            csv_folder = config.dataset_path,
            output_folder = config.output_path,
            word_limit = config.word_limit,
            min_word_count = config.min_word_count
        )