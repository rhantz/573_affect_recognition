import sys
import argparse
import re
import nltk
from nltk import SnowballStemmer
from nltk.corpus import stopwords

import pandas as pd


# The following list of stopwords can be used as a starting point if we want to manually compile our own list
# STOPS = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
#          'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
#          'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
#          'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
#          'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
#          'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
#          'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
#          'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
#          'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
#          'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
#          'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
#          'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 've', 'll']


def create_vectors(data: pd.DataFrame) -> list:
    """
    Given dataframe, returns a list of lists
    Args:
        data: dataframe
    Returns:
        vectors: list of feature vectors, each of which is a list of numbers (I think they will all be numbers!)
    """
    vectors = []
    # Do something here to create vectors from our data
    return vectors


def format_data(data: pd.DataFrame) -> list:
    """
    Given dataframe, returns a list of lists
    Args:
        data: dataframe
    Returns:
        x_and_y: list of lists
        x_and_y[0] is a list of feature vectors (ie, list of lists)
        x_and_y[1] is a list of emotion categories
    """
    y_list = data['emotion'].values.tolist()
    x_list = create_vectors(data)
    x_and_y = [x_list, y_list]
    return x_and_y


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


def process_text(text: str, lowercase=True, remove_punctuation=True, numbers='remove', remove_stop_words=False,
                 stem=False) -> str:
    """
    Given text, returns a processed version of the text
    Args:
        text: string
        *Currently function is set up with many default arguments - we may change this*
    Returns:
        text: processed version of input text according to the default arguments
    """
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
    text = ' '.join(text.split())
    return text


def get_word_counts(text: str) -> str:
    """
    Given text, returns a string of features
    Args:
        text: string
    Returns:
        feature_string: an alphabetized string of counts of words in input in format word1 count1 word2 count2
    """
    tokens = {}
    for item in text.split():
        if item in tokens:
            tokens[item] += 1
        else:
            tokens[item] = 1
    feature_string = ''
    for k in sorted(tokens.keys()):
        feature_string += k + ' ' + str(tokens[k]) + ' '
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


def get_args():
    """
    Returns: CLI arguments
    """

    parser = argparse.ArgumentParser(
        description="Reads in data")

    parser.add_argument(
        "--train_input_data_path", type=str, required=True,
        help="string path to initial training data from root directory")
    parser.add_argument(
        "--dev_input_data_path", type=str, required=False,
        help="string path to initial development data from root directory")
    parser.add_argument(
        "--test_input_data_path", type=str, required=False,
        help="string path to initial test data from root directory")

    return parser.parse_args(sys.argv[1:])


if __name__ == "__main__":
    arguments = get_args()
    all_datasets = get_datasets(arguments)
    updated_datasets = update_data(all_datasets)
    training_data_formatted = format_data(updated_datasets[0])


