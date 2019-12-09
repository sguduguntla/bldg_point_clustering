import pandas as pd
import numpy as np
import json
import yaml
import sys
import os

def word_matrix_to_df(wm, feat_names):
    """ Converts document-term matrix into a Pandas DataFrame

    :param wm: An array of featurized vectors
    :param feat_names: The array of words found in the corpus
    :return: Pandas DataFrame of the document-term matrix featurization of the corpus

    """
    # create an index for each row
    doc_names = ['Doc{:d}'.format(idx) for idx, _ in enumerate(wm)]
    df = pd.DataFrame(data=wm.toarray(), index=doc_names,
                      columns=feat_names)
    return df

def get_char_freq(df):
    """ Gets the frequency of every character in a Pandas DataFrame 
    
    :param df: Pandas DataFrame
    :return: Pandas Series of each character mapped to its respective frequency
    """

    data_str = "".join(df.values.flatten())

    return pd.Series([x for x in data_str]).value_counts()    

def read_file(filename, file_type="text"):
    """ Returns the data inside a file
    
    :param filename: Name of file (with extension) to read
    :param file_type: The type of the file being read (text, json, yaml)
    :return: String if file_type = "text", dictionary if file_type = "json" or "yaml"
    """

    my_path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(my_path, "..", filename)

    with open(os.path.abspath(path), "r") as f:
        if file_type == "json":
            return json.load(f)
        elif file_type == "yaml":
            return yaml.full_load(f)
        else:
            return f.read()

def write_file(filename, data, file_type="text"):
    """ Writes data to a file
    
    :param filename: Name of file (with extension) to read
    :param data: The data to be written into the file (String if file_type="text" or JSON if file_type="json" or "yaml")
    :param file_type: The type of the file being read (text, json, yaml)
    """
    my_path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(my_path, "..", filename)

    with open(os.path.abspath(path), "w") as f:
        if file_type == "json":
            json.dump(data, f)
        elif file_type == "yaml":
            yaml.dump(data, f)
        else:
            f.write(str(data))