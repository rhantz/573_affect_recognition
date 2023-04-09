"""
Word2Vec embedding vector module

    'from w2v_embeddings import W2V_Embeddings'

Creates a W2V_Embeddings object given a list of texts

    'processed_essays = data['essay_processed'].values.tolist()'
    'w2v = W2V_Embeddings(processed_essays)'

The following method calculates the centroid of an essay string

    'centroid = w2v.find_centroid(processed_essay)'

"""


import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize


class W2V_Embeddings:

    def __init__(self, all_essays: list):
        self.vocabulary = self.get_tokens(all_essays)
        self.model = self.train_model(self.vocabulary)

    @staticmethod
    def get_tokens(essays: list) -> list:
        """
        Given a list of strings, returns a list of lists of tokens
        Args:
            essays: list of essays in string format

        Returns:
            list of lists of tokens, in the format required by Word2Vec for training model
        """

        token_list = []
        for essay in essays:
            token_list.append(word_tokenize(essay))
        return token_list

    @staticmethod
    def train_model(tokens: list) -> Word2Vec:
        """
        Given a list of lists of tokens, trains a Word2Vec model
        Args:
            tokens: list of lists of tokens

        Returns:
             Word2Vec model
        """

        w2v_model = Word2Vec(sentences=tokens, min_count=1)
        return w2v_model

    def find_centroid(self, essay: str) -> list:
        """
        Given an essay, returns a vector representation of the essay
        Args:
            essay: string representation of an essay

        Returns:
             mean of vector representations of words in corpus
        """

        vecs = [self.model.wv[w] for w in essay if w in self.model.wv]
        vecs = [v for v in vecs if len(v) > 0]
        centroid = np.mean(vecs, axis=0)
        centroid = centroid.reshape(1, -1)
        return centroid.tolist()[0]
