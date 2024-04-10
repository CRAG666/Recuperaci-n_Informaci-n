# practica4.py
# Developer: Diego Cruz Aguilar
# Date: 09/04/2024
# Versión: 1.0

import math
import sys

sys.path.append("../Tarea3")

from practica3 import Document, TestIR, load_frequencies


class IRSystem4:
    def __init__(self, documents_frequencies) -> None:
        self.word_embeddings = self.__load_embeddings()
        self.documents_frequencies = documents_frequencies
        self.vocabulary = self.__get_vocabulary()
        self.documents_vectors = self.__vectorize_documents()

    def __load_embeddings(self):
        glove_file = "../dataset/GloVe/glove.42B.300d.txt"
        word_embeddings = {}
        with open(glove_file, "r", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = [float(val) for val in values[1:]]
                word_embeddings[word] = vector
        return word_embeddings

    def __get_vocabulary(self):
        vocabulary = {}
        for document in self.documents_frequencies:
            for word, frequency in document.frequencies.items():
                if vocabulary.get(word):
                    vocabulary[word] += 1
                else:
                    vocabulary[word] = 1
        return vocabulary

    def __vectorize_query(self, query: Document):
        query_vector = [0.0] * len(next(iter(self.word_embeddings.values())))
        num_words = 0
        for word in self.vocabulary:
            if word in query.frequencies and word in self.word_embeddings:
                word_vector = self.word_embeddings[word]
                query_vector = [x + y for x, y in zip(query_vector, word_vector)]
                num_words += 1
        if num_words != 0:
            query_vector = [x / num_words for x in query_vector]
        return query_vector

    def __vectorize_documents(self):
        documents_vectors = []
        for doc_freq in self.documents_frequencies:
            document_vector = [0.0] * len(next(iter(self.word_embeddings.values())))
            print(len(document_vector))
            num_words = 0
            print(doc_freq.frequencies.keys())
            for word in doc_freq.frequencies.keys():
                if word in self.word_embeddings:
                    word_vector = self.word_embeddings[word]
                    document_vector = [
                        x + y for x, y in zip(document_vector, word_vector)
                    ]
                    num_words += 1
            if num_words != 0:
                document_vector = [x / num_words for x in document_vector]
            documents_vectors.append(document_vector)
        return documents_vectors

    def __cosine_similarity(self, query, document):
        dot_product = sum(q * d for q, d in zip(query, document))
        norm_query = math.sqrt(sum(q**2 for q in query))
        norm_document = math.sqrt(sum(d**2 for d in document))
        if norm_query * norm_document == 0:
            return 0
        return dot_product / (norm_query * norm_document)

    def find(self, query):
        query = self.__vectorize_query(query)
        for doc_id, document_vector in enumerate(self.documents_vectors, start=1):
            similarity = self.__cosine_similarity(query, document_vector)
            if similarity > 0:
                yield doc_id, similarity


if __name__ == "__main__":
    import heapq

    test = TestIR("../dataset/TIME.REL")
    queries_frequencies = load_frequencies("../freq/queries", True)
    ir_system = IRSystem4(load_frequencies("../freq/documents"))
    with open("../result/retrieved_documents-GloVe.REL", "w") as output_file:
        for qu_id, query in enumerate(queries_frequencies[:10], start=1):
            output_line = f"Q{qu_id} "
            retrieved_documents_similarity = heapq.nlargest(
                10, ir_system.find(query), key=lambda x: x[1]
            )
            for doc_id, sim in retrieved_documents_similarity:
                output_line += f"D{doc_id} {sim} "
            retrieved_documents = [d for d, _ in retrieved_documents_similarity]
            precision, recall, f_measure, _ = test.evaluate(qu_id, retrieved_documents)
            output_line += f"P{precision} R{recall} F{f_measure}\n"
            print(output_line)
            output_file.write(output_line)
