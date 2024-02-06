# practica1.py
# Description: This program performs the preprocessing and analysis of the time dataset, with the purpose of obtaining the frequencies of the words contained in each document, and this process is applied to the queries.
# Developer: Diego Cruz Aguilar
# Date: 02/02/2024
# Versión: 2.0

from collections import Counter
from dataclasses import dataclass

import nltk
from nltk import word_tokenize
from nltk.corpus.reader import documents
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

# Ignore ALPHABET symbols
ALPHABET = [chr(i) for i in range(ord("A"), ord("Z") + 1)]
# For remove empty string
ALPHABET.append("")


def load_stopwords(stopwords_file: str) -> list[str]:
    """load stopwords file

    Args:
        stopwords_file: Path stopword file

    Returns: List to stopwords

    """
    with open(stopwords_file, "r") as file:
        stopwords = list()
        # Clean stopwords
        for word in file:
            w = word.strip()
            if w not in ALPHABET:
                stopwords.append(w.lower())
        return stopwords


# Struc for storage document info
@dataclass
class Document:
    Id: int
    Id_Psical: str
    Date: str
    Page: str
    Content: str


def get_metadata(title: str) -> tuple[str, str, str]:
    """Get metadata from tietle document

    Args:
        title: string title

    Returns: id, date and num page
    """
    page_index = title.find("PAGE")
    page = title[page_index:].split()
    id_date = title[:page_index].split(maxsplit=1)

    return id_date[0], id_date[1].rstrip(), page[1]


def load_documents(documents_path: str) -> list[Document]:
    """load and preprocessing documents

    Args:
        documents_path: Path form documents

    Returns: List of Documents

    """
    with open(documents_path, "r") as file:
        text = file.read()

    documents = []

    # Split docucuents from title
    fragments = text.split("*TEXT")[1:]

    for id, fragment in enumerate(fragments, start=1):
        # Split document for extract metadata
        lines = fragment.strip().split("\n")

        if lines:
            # Extract metadata
            id_psical, date, page = get_metadata(lines[0])
            content = " ".join(lines[1:])
            # Create new document
            document = Document(id, id_psical, date, page, content)
            documents.append(document)

    return documents


def load_queries(queries_path: str) -> list[str]:
    """Load and preprocessing queries

    Args:
        queries_path: Path from queries

    Returns: List of queries

    """
    with open(queries_path, "r") as file:
        text = file.read()

    queries = []

    fragments = text.split("*FIND")[1:]

    for id, fragment in enumerate(fragments, start=1):
        lines = fragment.strip().split("\n", maxsplit=1)
        if lines:
            queries.append(lines[1])

    return queries


class IRSystem1:
    """Class creating an information retrieval system

    Attributes:
        stop_words: List of stopwords
        documents: List of Documents
        wordnet_lemmatizer: Instance of WordNetLemmatizer
        ps: Instance of PorterStemmer algorihtm
    """

    def __init__(self, dataset_prefix_path: str) -> None:
        """Contructor

        Args:
            dataset_prefix_path: Relative Path of dataset files
        """
        self.stop_words = load_stopwords(dataset_prefix_path + ".STP")
        self.documents = load_documents(dataset_prefix_path + ".ALL")
        self.wordnet_lemmatizer = WordNetLemmatizer()
        self.ps = PorterStemmer()

    def preprocess_doc(self, text: str) -> list:
        """Preprocess text of documents or queries

        Args:
            text: Text for applty preprocessing

        Returns: List of words

        """
        word_tokens = word_tokenize(text.lower())

        preprocessed_tokens = [
            self.wordnet_lemmatizer.lemmatize(self.ps.stem(w))
            for w in word_tokens
            if w.isalnum() and w not in self.stop_words
        ]

        return preprocessed_tokens

    def extract_vocabulary(self):
        """Create file with words frequencies"""
        for document in self.documents:
            self.write_term_frecuency(
                f"{document.Id}-{document.Id_Psical}",
                Counter(self.preprocess_doc(document.Content)),
                "documents",
            )

    @staticmethod
    def write_term_frecuency(id: str, document_word_fecuency: Counter, doc_name: str):
        """Write frequencies in document

        Args:
            id: identifier for element
            document_word_fecuency: Dict with words and frequencies
            doc_name: Output file name
        """
        line = f"\nDoc{id}"
        for word, freq in document_word_fecuency.items():
            line += f" {word}-{freq}"
        with open(doc_name + ".FRQ", "a") as file:
            file.write(line)


if __name__ == "__main__":
    dataset_prefix_path = "./dataset/TIME"
    ir_system = IRSystem1(dataset_prefix_path)
    ir_system.extract_vocabulary()
    queries = load_queries(dataset_prefix_path + ".QUE")
    for id, query in enumerate(queries, start=1):
        ir_system.write_term_frecuency(
            f"{id}",
            Counter(ir_system.preprocess_doc(query)),
            "queries",
        )
