"""
Emotion enhanced Bag of Words vectorization module.

    `from emo_bow import EmoBoW`

Creates a EmoBoW vectorization object given a language ("eng" or "urdu") and a set of vocabulary

    `vocabulary = {set of all words in training data}`
    `emotion_vectorizer = EmoBoW(vocabulary, "eng")`

Has three options for vector creation given word count string (e.g. word_count_string = "love:3 happy:2 cat:1"):

Plain BoW vector:
    `bow_vector = emotion_vectorizer.bow_vector(word_count_string)`
    [0. 0. 0. 0. 1. 0. 2. 0. 0. 0. 0. 0. 0. 3.]

Emotion only vector: (six emotions "anger", "disgust", "fear", "joy", "sadness", "surprise")
    `emo_vector = emotion_vectorizer.emo_vector(word_count_string)`
    [0 0 0 5 0 0]

Emotion enhanced BoW vector: (emo_vector concatenated with bow_vector)
    `emo_bow_vector = emotion_vectorizer.emo_bow_vector(word_count_string)`
    [0. 0. 0. 5. 0. 0. 0. 0. 0. 0. 1. 0. 2. 0. 0. 0. 0. 0. 0. 3.]

"""

import pandas as pd
import numpy as np


class EmoBoW:

    # TODO: discuss stemming lexicon words and vocab words

    def __init__(self, vocab: set, language: str):
        self.emotion_lexicon = self.get_emotion_lexicon(language)
        self.vocab_indices = self.map_vocab_to_index(vocab)
        self.bow_vector_length = len(self.vocab_indices)

    @staticmethod
    def get_emotion_lexicon(language: str) -> dict:
        """
        Given a language, gets emotion lexicon as a word indexed dictionary of vectors
        Args:
            language: 'eng' or 'urdu' language string

        Returns:
            dictionary mapping word to emotion vector for emotions:
            "anger", "disgust", "fear", "joy", "sadness", "surprise"
        """

        f = "../data/emotion_lexicon/Urdu-NRC-EmoLex.txt"
        lexicon = pd.read_table(f, header=0)

        if language == "eng":
            word_column_id = "English Word"
        elif language == "urdu":
            word_column_id = "Urdu Word"
        else:
            raise ValueError("Please choose 'eng' or 'urdu' for the emotion lexicon language")

        emotions = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

        word = lexicon[word_column_id].tolist()
        emotion_vector = lexicon[emotions].values.tolist()

        return dict(zip(word, emotion_vector))

    @staticmethod
    def map_vocab_to_index(vocab: set) -> dict:
        """
        Given a set of vocabulary (the full training data vocabulary), sorts each vocab token alphabetically
        and maps to a unique index for later BoW vector creation

        Args:
            vocab: set of vocabulary

        Returns:
            vocab_indices: dictionary mapping each vocab token to an index

        """
        vocab_list = sorted(list(vocab))
        vocab_indices = {token: idx for idx, token in enumerate(vocab_list)}
        # "unk" is the token used for unknown tokens
        vocab_indices["unk"] = len(vocab_indices)
        return vocab_indices

    @staticmethod
    def map_word_to_count(word_count: str) -> dict:
        """
        Given a word_count string, splits string into dictionary mapping each word to its count in the
        current training instance.

        Args:
            word_count: string in format "a:3 also:4 ... zebra: 1"

        Returns:
            dictionary mapping each word to its count
            e.g. { a:3, also:4, zebra:1 }
        """
        return {word_count.split(":")[0]: int(word_count.split(":")[1]) for word_count in word_count.strip().split(" ")}

    def bow_vector(self, word_count: str) -> np.array:
        """
        Constructs Bag of Words vector from single training instance in format of word_count string

        Args:
            word_count: string in format "a:3 also:4 ... zebra: 1"

        Returns:
            Bag of Words vector
        """
        word_count_mapping = self.map_word_to_count(word_count)

        vector = np.zeros(self.bow_vector_length)

        for word, count in word_count_mapping.items():
            if word in self.vocab_indices:
                index = self.vocab_indices[word]
            else:
                # word is unknown if not in the vocab
                index = self.vocab_indices["unk"]
            vector[index] = count

        return vector

    def emo_vector(self, word_count:str) -> np.array:
        """
        Constructs emotion vector from single training instance in format of word_count string

        Args:
            word_count: string in format "a:3 also:4 ... zebra: 1"

        Returns:
            emotion vector where each index represents a different emotion and each value represents the
            number of words with that emotion in the training instance
        """

        word_count_mapping = self.map_word_to_count(word_count)

        vectors = []

        for word, count in word_count_mapping.items():
            # Only worry about words in the emotion lexicon
            if word in self.emotion_lexicon:
                vector = np.array(self.emotion_lexicon[word])
                vectors.append(vector*count)

        return np.sum(vectors, axis=0)

    def emo_bow_vector(self, word_count: str) -> np.array:
        """
        Concatenates emotion and Bag of Words vector for a single training instance in format of word_count string
        to create an emotion enhanced Bag of Words vector.

        Args:
            word_count: string in format "a:3 also:4 ... zebra: 1"

        Returns:
            emotion enhanced Bag of Words vector

        """
        emotion_vector = self.emo_vector(word_count)
        bow_vector = self.bow_vector(word_count)
        return np.concatenate((emotion_vector, bow_vector))
