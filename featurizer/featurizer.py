import numpy as np 
import pandas as pd
from bldg_point_clustering.helper import word_matrix_to_df, read_file
from bldg_point_clustering.tokenizer import arka_tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

class Featurizer:

    """ Creates a Featurizer object instance.

    :param filename: The name of the file containing the data to be featurized (Excluding file extension)
    :param corpus: The Pandas Series of the strings to be clustered

    """

    __heuristics = read_file("heuristics.yaml", file_type="yaml")

    def __init__(self, filename, corpus):
        
        assert filename and filename in self.__heuristics, "Invalid filename."
        assert isinstance(corpus, pd.Series), "Corpus must be of type Pandas Series"

        self.filename = filename
        self.corpus = self.__apply_heuristics(corpus.dropna())
        self.__word_matrix_df = None
    
    def bag_of_words(self, min_freq=1, max_freq=1.0, stop_words=None, tfidf=False): 
        """ Returns feature vectors based on the bag of words model, with each string
        tokenized using the arka tokenizer (Look in tokenizers for more information).

        :param min_freq: Minimum frequency of a word to include in the featurization (float from 0.0 to 1.0)
        :param max_freq: Maximum frequency of a word to include in the featurization (float from 0.0 to 1.0)
        :param stop_words: Array of stop words all of which will be removed from the resulting tokens.
        :param tfidf: Boolean indicating whether to use term frequency–inverse document frequency (TFIDF) model
        :return: Pandas DataFrame of the document-term matrix featurization of the corpus (Featurized DataFrame)
        """

        if tfidf:
            vectorizer = TfidfVectorizer(min_df=min_freq, stop_words=stop_words,  max_df=max_freq, tokenizer=arka_tokenizer, use_idf=True)
        else:
            vectorizer = CountVectorizer(min_df=min_freq, stop_words=stop_words, max_df=max_freq, tokenizer=arka_tokenizer)
        
        cvec = vectorizer.fit_transform(self.corpus)
        vectors = cvec.todense()
        
        # retrieve the terms found in the corpora
        tokens = vectorizer.get_feature_names()
        
        # create a dataframe from the matrix
        self.__word_matrix_df = word_matrix_to_df(cvec, tokens)

        feature_vectors = np.array(vectors)

        featurized_df = pd.DataFrame(feature_vectors.tolist()).fillna(0)

        return featurized_df
        
    def arka(self):
        """ Returns feature vectors based on the arka thesis model.

        :return: Pandas DataFrame of the document-term matrix featurization of the corpus
        """
                            
        def featurize_point(point):
            f_vector = []
            codes = {
                "alpha": 1,
                "numeric": 2,
                "special char": 3
            }

            consecutive = dict.fromkeys(["alpha", "numeric", "special char"], False)

            for char in point:
                if char.isalpha():
                    if not consecutive["alpha"]:
                        f_vector.append(codes["alpha"])
                        consecutive["alpha"] = True
                        consecutive["numeric"] = False
                        consecutive["special char"] = False
                elif char.isdigit():
                    if not consecutive["numeric"]:
                        f_vector.append(codes["numeric"])
                        consecutive["alpha"] = False
                        consecutive["numeric"] = True
                        consecutive["special char"] = False
                else:
                    if not consecutive["special char"]:
                        f_vector.append(codes["special char"])
                        consecutive["alpha"] = False
                        consecutive["numeric"] = False
                        consecutive["special char"] = True
        
            return np.array(f_vector)
        
        self.__word_matrix_df = None

        feature_vectors = self.corpus.apply(lambda x: featurize_point(x))

        featurized_df = pd.DataFrame(feature_vectors.tolist()).fillna(0)
        
        return featurized_df
    
    def get_word_matrix_df(self):
        """ Gets the Bag of Words document-term featurization matrix
        
        :return: Pandas DataFrame of Bag of Words document-term featurization matrix
        """

        return self.__word_matrix_df
    
    def __apply_heuristics(self, corpus):
        """ Applies the heuristics in heuristics.yaml by removing all the instances of the heuristic regular expressions.

        :param corpus: Pandas Series of strings to apply heuristics on
        """

        if self.filename in self.__heuristics:
            heuristic_regexps = [r"" + h for h in self.__heuristics[self.filename]]

            for exp in heuristic_regexps:
                corpus = corpus.str.replace(exp, '')
        
        return corpus
