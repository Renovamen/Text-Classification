import argparse
import yaml

class Config:
    """Convert a ``dict`` into a ``Class``"""
    def __init__(self, entries: dict = {}):
        for k, v in entries.items():
            if isinstance(v, dict):
                self.__dict__[k] = Config(v)
            else:
                self.__dict__[k] = v

def load_config(file_path: str) -> dict:
    """
    Load configuration from a YAML file

    Parameters
    ----------
    file_path : str
        Path to the config file (in YAML format)

    Returns
    -------
    config : dict
        Configuration settings
    """
    f = open(file_path, 'r', encoding = 'utf-8')
    config = yaml.load(f.read(), Loader = yaml.FullLoader)
    return config

def parse_opt() -> Config:
    parser = argparse.ArgumentParser()
    # config file
    parser.add_argument(
        '--config',
        type = str,
        default = 'configs/ag_news/han.yaml',
        help = 'path to the configuration file (yaml)'
    )
    args = parser.parse_args()
    config_dict = load_config(args.config)
    config = Config(config_dict)

    return config
