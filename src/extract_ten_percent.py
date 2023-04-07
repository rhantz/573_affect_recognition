import sys
import argparse
import pandas as pd

from sklearn.model_selection import train_test_split


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
        description="Extracts 10% of total data from training data")

    parser.add_argument(
        "--train_input_data_path", type=str, required=True,
        help="string path to initial training data from root directory")
    parser.add_argument(
        "--dev_input_data_path", type=str, required=False,
        help="string path to initial development data from root directory")
    parser.add_argument(
        "--test_input_data_path", type=str, required=False,
        help="string path to initial test data from root directory")
    parser.add_argument(
        "--train_output_data_path", type=str, required=True,
        help="string path to new training data from root directory")
    parser.add_argument(
        "--test_output_data_path", type=str, required=True,
        help="string path to new test data from root directory")
    parser.add_argument(
        "--data_row_id_column", type=str, required=True,
        help="string name of column for unique data row identifier")
    parser.add_argument(
        "--y_label_column", type=str, required=True,
        help="string name of column for y label")

    return parser.parse_args(sys.argv[1:])


if __name__ == "__main__":

    arguments = get_args()
    all_datasets = get_datasets(arguments)

    initial_train_data = all_datasets[0]

    # determines test_size
    num_data_rows = sum([len(dataset) for dataset in all_datasets])
    ten_percent_of_total = round(num_data_rows * .1, 0)
    test_size = ten_percent_of_total/len(initial_train_data)

    # extracts new train and test data
    # random_state is set to 1 for reproducibility
    X = initial_train_data[arguments.data_row_id_column].tolist()
    y = initial_train_data[arguments.y_label_column].tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)

    # create and format new train and test data as .tsv files
    new_train_data = initial_train_data[initial_train_data[arguments.data_row_id_column].isin(X_train)]
    new_test_data = initial_train_data[initial_train_data[arguments.data_row_id_column].isin(X_test)]
    new_train_data.to_csv(arguments.train_output_data_path, sep="\t")
    new_test_data.to_csv(arguments.test_output_data_path, sep="\t")










