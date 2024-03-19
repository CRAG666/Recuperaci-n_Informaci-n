# practica2.py
# Developer: Diego Cruz Aguilar
# Date: 18/03/2024
# Versión: 1.0

import math
from dataclasses import dataclass
from typing import Union


@dataclass
class Document:
    id: str
    frequencies: dict[str, int]


@dataclass
class DocumentFrequency(Document):
    id_psical: str


def load_frequencies(
    frequencies_name: str, query=False
) -> Union[list[DocumentFrequency], list[Document]]:
    documents_frequencies = []
    with open(f"{frequencies_name}.FRQ", "r") as file:
        for line in file:
            words_frequency = line.split()
            if not words_frequency:
                break
            if query:
                id = words_frequency[0]
                doc = Document(id=id, frequencies={})
            else:
                id, id_psical = words_frequency[0].split("-")
                doc = DocumentFrequency(id=id, id_psical=id_psical, frequencies={})
            for word_frequency in words_frequency[1:]:
                word, frquency = word_frequency.split("-")
                doc.frequencies[word] = int(frquency)
            documents_frequencies.append(doc)
    return documents_frequencies


class TestIR:
    def __init__(self, file_results: str) -> None:
        self.file_results = file_results
        self.relevant_documents_per_query = self.__load_expected_result()

    def __load_expected_result(self):
        with open(self.file_results, "r") as file:
            lines = file.readlines()
        data = {}
        for line in lines:
            words = line.split()
            if len(words) == 0:
                continue
            key = int(words[0])
            data[key] = [int(x) for x in words[1:]]
        return data

    def average_precision(self, relevant_documents, retrieved_documents):
        relevants = 0
        precision = 0.0
        for i, p in enumerate(retrieved_documents):
            if p in relevant_documents:
                relevants += 1
                precision += relevants / (i + 1)
        if relevants == 0:
            return 0.0
        else:
            return precision / relevants

    def evaluate(self, id_query: int, retrieved_documents: list):
        relevant_documents = self.relevant_documents_per_query[id_query]
        true_positives = len(set(relevant_documents) & set(retrieved_documents))
        precision = true_positives / len(retrieved_documents)
        recall = true_positives / len(relevant_documents)
        if precision + recall == 0:
            f_measure = 0
        else:
            f_measure = 2 * (precision * recall) / (precision + recall)
        ap = self.average_precision(relevant_documents, retrieved_documents)
        return precision, recall, f_measure, ap


class IRSystem2:
    def __init__(self, documents_frequencies) -> None:
        self.documents_frequencies = documents_frequencies
        self.vocabulary = self.__get_vocabulary()
        self.documents_vectors = self.__vectorize_documents()

    def __get_vocabulary(self):
        vocabulary = {}
        for document in self.documents_frequencies:
            for word, frequency in document.frequencies.items():
                if vocabulary.get(word):
                    vocabulary[word] += 1
                else:
                    vocabulary[word] = 1
        return vocabulary

    def __vectorize_documents(self):
        documents_vectors = []
        num_doc = len(self.documents_frequencies)
        for doc_freq in self.documents_frequencies:
            document_vector = {}
            size_document = sum(doc_freq.frequencies.values())
            for word, frecuency in self.vocabulary.items():
                tf = doc_freq.frequencies.get(word, 0) / size_document
                idf = math.log(num_doc / (1 + frecuency))
                document_vector[word] = tf * idf
            documents_vectors.append(document_vector)
        return documents_vectors

    def __vectorize_query(self, query: Document):
        query_vector = {}
        for word in self.vocabulary:
            if word in query.frequencies:
                query_vector[word] = 1
            else:
                query_vector[word] = 0
        return query_vector

    def __cosine_similarity(self, query, document):
        dot_product = sum(query[word] * document[word] for word in query)
        norm_query = math.sqrt(sum(query[word] ** 2 for word in query))
        norm_document = math.sqrt(sum(document[word] ** 2 for word in document))
        if norm_query * norm_document == 0:
            return 0
        return dot_product / (norm_query * norm_document)

    def rocchio_feedback(
        self, query, relevant_docs, irrelevant_docs, alpha, beta, gamma
    ):
        query_vector = self.__vectorize_query(query)
        relevant_vectors = [
            self.documents_vectors[doc_id - 1] for doc_id in relevant_docs
        ]
        irrelevant_vectors = [
            self.documents_vectors[doc_id - 1] for doc_id in irrelevant_docs
        ]
        updated_query_vector = {}
        for word in self.vocabulary:
            relevant_term = sum(doc_vector[word] for doc_vector in relevant_vectors)
            irrelevant_term = sum(doc_vector[word] for doc_vector in irrelevant_vectors)
            updated_query_vector[word] = (
                alpha * query_vector[word]
                + beta * (relevant_term / len(relevant_docs))
                - gamma * (irrelevant_term / len(irrelevant_docs))
            )
        return updated_query_vector

    def find(self, query, vector=False):
        if not vector:
            query = self.__vectorize_query(query)
        for doc_id, document_vector in enumerate(self.documents_vectors, start=1):
            similarity = self.__cosine_similarity(query, document_vector)
            if similarity > 0:
                yield doc_id, similarity


if __name__ == "__main__":
    import heapq

    test = TestIR("../dataset/TIME.REL")
    queries_frequencies = load_frequencies("../freq/queries", True)
    ir_system = IRSystem2(load_frequencies("../freq/documents"))
    with open("../result/retrieved_documents_rocchio.REL", "w") as output_file:
        for qu_id, tmp_query in enumerate(queries_frequencies[:10], start=1):
            output_line = f"Q{qu_id} "
            rd = ir_system.find(tmp_query)
            rdc = list(map(lambda x: x[0], sorted(rd, key=lambda x: x[1])))
            relevant_docs = rdc[:3]
            irrelevant_docs = rdc[3:]
            # relevant_docs = set(test.relevant_documents_per_query.get(qu_id, [])) & rdc
            # irrelevant_docs = rdc - relevant_docs
            # relevant_docs = test.relevant_documents_per_query.get(qu_id, [])
            # irrelevant_docs = list(
            #     set(range(1, len(ir_system.documents_vectors) + 1)) - set(relevant_docs)
            # )
            query = ir_system.rocchio_feedback(
                tmp_query,
                relevant_docs,
                irrelevant_docs,
                alpha=1,
                beta=0.75,
                gamma=0.25,
            )
            retrieved_documents_similarity = heapq.nlargest(
                10, ir_system.find(query, True), key=lambda x: x[1]
            )
            for doc_id, sim in retrieved_documents_similarity:
                output_line += f"D{doc_id} {sim} "
            retrieved_documents = [d for d, _ in retrieved_documents_similarity]
            precision, recall, f_measure, ap = test.evaluate(qu_id, retrieved_documents)
            output_line += f"P{precision} R{recall} F{f_measure} AP{ap}\n"
            print(output_line)
            output_file.write(output_line)
