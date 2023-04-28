import re
import nltk
from nltk import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
import pandas as pd

# The following lines of code are necessary for method handle_negation
nltk.download('wordnet')
from spacy_wordnet.wordnet_annotator import WordnetAnnotator
spacy.cli.download("en_core_web_sm")
nltk.download("omw")
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("spacy_wordnet", after="tagger")


def update_data(datasets: list) -> list:
    """
    Given list of dataframes, returns a list of dataframes with additional columns
    Args:
        datasets: list of dataframes
    Returns:
        updated_datasets: list of dataframes
    """
    updated_datasets = []
    for df in datasets:
        df['essay_processed'] = df['essay'].apply(process_text)
        df['word_counts'] = df['essay_processed'].apply(get_word_counts)
        updated_datasets.append(df)
    return updated_datasets


def process_text(text: str, remove_neg=True, tokenize=False, lowercase=True, remove_punctuation=True, numbers='remove', remove_stop_words=False,
                 stem=False) -> str:
    """
    Given text, returns a processed version of the text
    Args:
        text: string
        *Currently function is set up with many default arguments - we may change this*
    Returns:
        text: processed version of input text according to the default arguments
    """
    if tokenize:
        text = ' '.join(word_tokenize(text))
    if lowercase:
        text = text.lower()
    if numbers == 'replace':
        text = re.sub('[0-9]+', 'NUM', text)
    if numbers == 'remove':
        text = re.sub('[0-9]+', ' ', text)
    if remove_punctuation:
        text = re.sub(r'[^\sA-Za-z0-9À-ÖØ-öø-ÿЀ-ӿ/]', '', text)
    if stem:
        snow_stemmer = SnowballStemmer(language='english')
        text = ' '.join([snow_stemmer.stem(word) for word in text.split()])
    if remove_stop_words:
        stops = set(stopwords.words('english'))
        text = ' '.join([word for word in text.split() if word not in stops])
    if remove_neg:
        text = handle_negation(text)
    text = ' '.join(text.split())
    return text

def handle_negation(text: str) -> str:
    """
    Given text, returns an updated version of text in which negated words are replaced with antonyms
    Args:
        text: string
    Returns:
        text: string, with negated words replaced with antonyms
    """
    doc = nlp(text)
    negations = set()
    negated_words = set()
    all_tokens = []
    token_index = 0
    for token in doc:
        all_tokens.append(token)
        if token.dep_ == 'neg':
            negations.add(token)
            negated_words.add(token.head)
        token_index += 1
    updated_words = []
    for token in all_tokens:
        if token in negations:
            continue
        elif token in negated_words:
            lemmas = token._.wordnet.lemmas()
            if lemmas:
                for lemma in lemmas:
                    antonyms = lemma.antonyms()
                    if antonyms:
                        antonym = antonyms[0].name()
                        updated_words.append(antonym)
                        break
        else:
            updated_words.append(token.orth_)
    updated_text = ' '.join(w for w in updated_words)
    return updated_text


def get_word_counts(text: str) -> str:
    """
    Given text, returns a string of word counts
    Args:
        text: string
    Returns:
        feature_string: an alphabetized string of words from input formatted word1:count1 word2:count2 etc
    """
    tokens = {}
    for item in text.split():
        if item in tokens:
            tokens[item] += 1
        else:
            tokens[item] = 1
    feature_string = ''
    for k in sorted(tokens.keys()):
        feature_string += k + ':' + str(tokens[k]) + ' '
    return feature_string


def get_data(f: str) -> pd.DataFrame:
    """
    Given path to data .tsv, gets data as a dataframe
    Args:
        f: path to data .tsv
    Returns:
        data: data as dataframe
    """

    data = pd.read_table(f, header=0)
    return data


def get_datasets(args) -> list:
    """
    Given input dataset paths provided in arguments, creates list of dataframe datasets.
    Only args.train_input_data is required.
    Args:
        args: CLI arguments
    Returns:
        datasets: list of dataframe datasets. training data will be given first.
    """

    train_data = get_data(args.train_input_data_path)
    datasets = [train_data]

    if args.dev_input_data_path:
        dev_data = get_data(args.dev_input_data_path)
        datasets.append(dev_data)

    if args.test_input_data_path:
        test_data = get_data(args.test_input_data_path)
        datasets.append(test_data)

    return datasets



