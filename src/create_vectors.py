import pandas as pd


from w2v_embeddings import W2V_Embeddings, Pretrained_Embeddings
from emo_bow import EmoBoW


def make_vectors(args, datasets: list) -> list:
    """
    Given input arguments and list of datasets, creates list of vectors and gold labels
    Args:
        args: CLI arguments
        datasets: list of DataFrames
    Returns:
         formatted data: list of lists - in each list, list[0] are feature vectors and list [1] are gold labels
    """
    vector_type = args.vector_type
    try:
        if args.urdu:
            if vector_type == 'pretrained':
                raise ValueError(
                    "There are no pretrained models available for Urdu data")
            elif vector_type == 'w2v':
                            raise ValueError(
                    "w2v is not a valid selection for Urdu data")
            else:
                formatted_data = build_vectors_emobow(vector_type, datasets, 'urdu')
                return formatted_data
    except AttributeError:
        if vector_type == 'pretrained':
            model_name = args.pretrained_model
            formatted_data = build_vectors_pretrained(datasets, model_name)
        elif vector_type == 'w2v':
            formatted_data = build_vectors_w2v(datasets)
        else:
            formatted_data = build_vectors_emobow(vector_type, datasets, 'eng')
        return formatted_data


def build_vectors_pretrained(datasets: list, model_name: str) -> list:
    """
    Given input dataframe, creates vectors for the dataframes
    Args:
        datasets: list of DataFrames
        model_name: string name of pretrained gensim model
    Returns:
         all_x_y: list of lists - in each list, list[0] are feature vectors and list [1] are gold labels
    """
    model = Pretrained_Embeddings(model_name)
    all_x_y = []
    for i in range(len(datasets)):
        data = datasets[i]
        essays = data['essay_processed'].values.tolist()
        labels = data['emotion'].values.tolist()
        vectors = []
        for essay in essays:
            vectors.append(model.find_centroid(essay))
        all_x_y.append([vectors, labels])
    return all_x_y


def build_vectors_w2v(datasets: list) -> list:
    """
    Given input dataframe, creates vectors for the dataframes
    Args:
        datasets: list of DataFrames
    Returns:
         all_x_y: list of lists - in each list, list[0] are feature vectors and list [1] are gold labels
    """
    # First we need to train the model on the training data, and build vectors for the training data
    train_data = datasets[0]
    train_essays = train_data['essay_processed'].values.tolist()
    train_labels = train_data['emotion'].values.tolist()
    w2v = W2V_Embeddings(train_essays)
    train_vectors = []
    for essay in train_essays:
        train_vectors.append(w2v.find_centroid(essay))
    train_x_y = [train_vectors, train_labels]
    all_x_y = [train_x_y]

    # Now the model has been trained, we create vectors for the dev and/or test data
    for i in range(1, len(datasets)):
        data = datasets[i]
        essays = data['essay_processed'].values.tolist()
        labels = data['emotion'].values.tolist()
        vectors = []
        for essay in essays:
            vectors.append(w2v.find_centroid(essay))
        all_x_y.append([vectors, labels])
    return all_x_y


def build_vectors_emobow(vector_type: str, datasets: list, lang: str):
    """
    Given input dataframe, creates vectors for the dataframes
    Args:
        vector_type: specifies plain BoW vector, emotion-only vector, or emotion enhanced BoW vector
        datasets: list of DataFrames
    Returns:
         all_x_y: list of lists - in each list, list[0] are feature vectors and list [1] are gold labels
    """
    vocabulary = get_vocabulary(datasets[0])
    emotion_vectorizer = EmoBoW(vocabulary, lang)
    all_x_y = []
    for i in range(len(datasets)):
        data = datasets[i]
        word_counts = data['word_counts'].values.tolist()
        labels = data['emotion'].values.tolist()
        vectors = []
        for word_count_string in word_counts:
            if vector_type == "bow_only":
                vectors.append(emotion_vectorizer.bow_vector(word_count_string))
            elif vector_type == "emo_only":
                vectors.append(emotion_vectorizer.emo_vector(word_count_string))
            else:
                vectors.append(emotion_vectorizer.emo_bow_vector(word_count_string))
        all_x_y.append([vectors, labels])
    return all_x_y


def get_vocabulary(data: pd.DataFrame) -> set:
    """
    Given input dataframe, creates a list of all words in the vocabulary
    Args:
        data: DataFrame of training data
    Returns:
         vocabulary: set of all words in the vocabulary
    """
    vocab_set = set()
    for row in data.iterrows():
        text = data.loc[row[0], 'essay_processed']
        tokens = text.split()
        for token in tokens:
            vocab_set.add(token)
    return vocab_set
