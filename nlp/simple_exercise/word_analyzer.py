"""
Please install libraries using requirements.txt.
Please also ensure the following is run prior to first run:

    nltk.download('stopwords')
    nltk.download('wordnet')

NOTES: I've notice random "?" are present in some of the files...

"""

import multiprocessing
import os
import re
import string
import warnings
from collections import Counter
from copy import deepcopy

import nltk
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

data_path = "{}/data/".format(os.path.abspath(os.curdir))


def find_whole_word(word, text):
    """
    'in' not sufficient, require regex for multi-char word boundaries

    :param word: word to match
    :param text: sentence to search
    :return: boolean
    """
    return re.findall(r'\b({0})\b'.format(word), text, flags=re.IGNORECASE)


def source_from_local_txt(file_name, path=data_path):
    """
    Serialize source file into a text string.

    :param file_name: Name of the file
    :param path: abs path of the file.
    :return: str - string of the file.
    """

    with open(path + file_name) as fh:
        raw_text = fh.read()

    return raw_text


def parse_and_tokenize(raw_text, file_name, remove_stop_words=False):
    """
    Function will parse the raw text files in the path provided. - "path" default's to "./data" directory.

    :param raw_text: text string of the entire document
    :param file_name: document name
    :param remove_stop_words: common words in language which do not have much meaning.

    :return: words_dict = {
                    "word1": {
                        "word_count": 34,
                        "sentences": ["sentence 1", "sentence 2"],
                        "documents": ["doc1.txt", "doc2.txt"]
                    },

                    "word2": {
                        "word_count": 34,
                        "sentences": ["sentence 3", "sentence 4"],
                        "documents": ["doc3.txt", "doc2.txt"]
                    }
                }
    """

    words_dict = {}

    # Remove weird chars - “”‘’
    text = raw_text.translate(str.maketrans("", "", "“”‘’"))

    # Replacing newline with white space
    text = text.lower().replace("\n", " ")

    # string.Counter does not break up compound words - e.g "on-the-job"
    # generally accepted rule is that a compound word is always treated as a single word
    word_counts = Counter([word.strip(string.punctuation) for word in text.split() if word not in string.punctuation])

    # Optional remove stop words (very common language words which can be non-informative)
    stop_words = nltk.corpus.stopwords.words('english') if remove_stop_words else []

    for stop_word in stop_words:
        word_counts.pop(stop_word, None)

    # Senetences are split over periods, exclamations and question marks. Noticed "?" mid-sentences
    all_sentences = re.split("[.?!]", raw_text)

    # Build nested dict - see docstring.
    for word in word_counts.keys():

        sentences = [sentence.strip() for sentence in all_sentences
                     if find_whole_word(word, sentence.lower())]

        words_dict[word] = dict(
            word_count=word_counts[word],
            sentences=sentences,
            documents=[file_name]
        )

    return words_dict


def arrange_common_words(all_words):
    """
    Construct a Pandas dataframe from words dictionary
    :param all_words: dictionary of words that contains word_count, sentences and documents for a word.
    :return:
    """

    word_details = all_words.values()
    columns = ("word_count", "documents", "sentences")

    # Arrange data in a Numpy array to be passed to a Pandas DF - shape (num_of_words, columns)
    data = np.array([np.array([detail[column] for detail in word_details]) for column in columns]).T

    # Pandas Dataframe
    df = pd.DataFrame(
        data,
        all_words.keys(),
        columns
    )

    return df.sort_values("word_count", ascending=False)


def _stem_lemm(df, func):
    """
    Helper function that updates the DataFrame with an nltk stemming/lemming function

    :param df:
    :param func:
    :return:
    """

    new_df = pd.DataFrame(columns=df.columns)

    for word in df.index:

        stem_lemm_word = func(word)

        if stem_lemm_word not in new_df.index:
            new_df.ix[stem_lemm_word] = deepcopy(df.ix[word])
        else:
            new_df.ix[stem_lemm_word] = new_df.ix[stem_lemm_word] + df.ix[word]

    return new_df


def stemmatize(df):
    """
    Function that applies the SnowballStemmer on all entries in the DF.
    Combines results for words that have the same root.

    :param df:
    :return:
    """
    stemmer = nltk.stem.SnowballStemmer('english')
    return _stem_lemm(df, stemmer.stem)


def lemmatize(df):
    """
    Function that applies a Lemming function on all entries in the DF.
    Combines results for words that have the same root.
    :param df:
    :return:
    """
    lemma = nltk.wordnet.WordNetLemmatizer()
    return _stem_lemm(df, lemma.lemmatize)


def write_words_df(df, filename):
    """
    Function to write df results in the requested format
    :param filename: filename to save the output
    :param df: words data frame
    :return:
    """

    def make_template():
        output_template = "Word(#)\t\tDocuments\tSentences containing the word:\n"
        output_template += "{0}\t\t{1}\n".format(word, ", ".join(doc for doc in documents)) + "\n" + "-" * 70 + "\n"

        for sentence in sentences:
            output_template += sentence + "\n\n"

        output_template += "-" * 70 + "\n"
        return output_template

    with open(filename + ".txt", "w") as fh:

        fh.write("-" * 70 + "\n")

        for word in df.index:
            documents = df.ix[word]["documents"]
            sentences = df.ix[word]["sentences"]
            fh.write(make_template())

    # Tab delimited CSV which is a more simpler format than above text format
    with open(filename + "_tab.txt", "w") as fh:
        df.to_csv(fh, header=list(df.columns), index=list(df.index), sep='\t', mode='a')


def main():

    # Multiprocessing file parsing & tokenization for the multiple source files
    file_names = os.listdir(data_path)
    file_as_strings = [source_from_local_txt(file_name) for file_name in file_names]
    args = [arg for arg in zip(file_as_strings, file_names, [False] * len(file_names))]

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.starmap(parse_and_tokenize, args)

    # Amalgamate all the results from the different processes into a single dict
    all_words_dict = results[0]

    for result in results[1:]:

        for this_word, this_word_info in result.items():
            existing_word_count = all_words_dict[this_word]["word_count"] if all_words_dict.get(this_word) else 0
            existing_sentences = all_words_dict[this_word]["sentences"] if all_words_dict.get(this_word) else []
            existing_documents = all_words_dict[this_word]["documents"] if all_words_dict.get(this_word) else []

            all_words_dict[this_word] = dict(
                word_count=this_word_info["word_count"] + existing_word_count,
                sentences=this_word_info["sentences"] + existing_sentences,
                documents=this_word_info["documents"] + existing_documents
            )

    top_words_df = arrange_common_words(all_words_dict)

    # Write DF to file in requested format
    write_words_df(top_words_df, "top_words")

    # Print Top 20 words DataFrame & write to file
    top_20 = top_words_df.head(n=20)
    write_words_df(top_words_df, "top_20")

    # Add a new columnn to the DataFrame showing the number of sentences each word appeared in
    top_20["num_of_sentences"] = top_20["sentences"].apply(len)
    print(top_20)

    # Lemmatized DF - grouping together the different inflected forms of a word
    # so they can be analysed as a single item.
    top_20_lemmatized = lemmatize(top_20)
    print(top_20_lemmatized)
    write_words_df(top_words_df, "top_20_lemmatized")

    # Stemming (reducing inflected/derived words to their stem, base or root form)
    # e.g fishing, fished, and fisher to the stem fish
    top_20_stemmatized = stemmatize(top_20)
    print(top_20_stemmatized)
    write_words_df(top_words_df, "top_20_stemmatized")


if __name__ == "__main__":
    main()
