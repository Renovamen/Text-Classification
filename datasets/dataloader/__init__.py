from datasets.dataloader import sentence, document

'''
load data

input param:
    config (Class): config settings
    split: 'trian' / 'test'
    build_vocab: whether to build vocabulary (True / False)
                 only makes sense when split = 'train'
'''
def load_data(config, split, build_vocab = True):
    if config.model_name in ['han']:
        return document.load_data(config, split, build_vocab)
    else:
        return sentence.load_data(config, split, build_vocab)