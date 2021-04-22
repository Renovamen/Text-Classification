def get_clean_text(text: str) -> str:
    """
    Preprocess text for being used in the model, including lower-casing,
    standardizing newlines and removing junk.

    Parameters
    ----------
    text : str
        A string to be cleaned

    Returns
    -------
    clean_text : str
        String after being cleaned
    """
    if isinstance(text, float):
        return ''

    clean_text = text.lower().replace('<br />', '\n').replace('<br>', '\n').replace('\\n', '\n').replace('&#xd;', '\n')
    return clean_text
