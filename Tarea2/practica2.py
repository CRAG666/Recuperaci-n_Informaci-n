# practica2.py
# Developer: Diego Cruz Aguilar
# Date: 25/02/2024
# Versión: 3.0

import csv
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
        for linea in file:
            palabras = linea.split()
            if not palabras:
                break
            if query:
                id = palabras[0]
                doc = Document(id=id, frequencies={})
            else:
                id, id_psical = palabras[0].split("-")
                doc = DocumentFrequency(id=id, id_psical=id_psical, frequencies={})
            for palabra_valor in palabras[1:]:
                palabra, valor = palabra_valor.split("-")
                doc.frequencies[palabra] = int(valor)
            documents_frequencies.append(doc)
    return documents_frequencies


class IRSystem2:
    def __init__(self, documents_frequencies) -> None:
        self.documents_frequencies = documents_frequencies
        self.vocabulary = self.__get_vocabulary()
        self.documents_vectors = self.__vectoricer_documents()

    def __get_vocabulary(self):
        vocabulary = dict()
        for document in self.documents_frequencies:
            for word, _ in document.frequencies.items():
                if vocabulary.get(word):
                    vocabulary[word] += 1
                else:
                    vocabulary[word] = 1

        with open("frequencies.csv", "w", newline="") as csv_file:
            escritor_csv = csv.DictWriter(csv_file, fieldnames=vocabulary.keys())
            escritor_csv.writeheader()
            escritor_csv.writerow(vocabulary)
        return vocabulary

    def __vectoricer_documents(self):
        documents_vectors = [self.vocabulary.copy() for _ in self.documents_frequencies]
        for index, document in enumerate(documents_vectors):
            for word, frequency in document.items():
                if word in self.documents_frequencies[index].frequencies:
                    documents_vectors[index][word] = (
                        self.documents_frequencies[index].frequencies[word]
                        / self.vocabulary[word]
                    )
                else:
                    documents_vectors[index][word] = 0
        return documents_vectors

    def vectoricer_query(self, query: Document):
        query_vector = self.vocabulary.copy()
        for word, frequency in query_vector.items():
            if word in query.frequencies:
                query_vector[word] = 1
            else:
                query_vector[word] = 0
        return query_vector

    def cosine_similarity(self, query_vector, document_vector):
        dot_product = sum(
            query_vector[term] * document_vector.get(term, 0) for term in query_vector
        )
        query_norm = math.sqrt(sum(value**2 for value in query_vector.values()))
        document_norm = math.sqrt(sum(value**2 for value in document_vector.values()))
        if query_norm == 0 or document_norm == 0:
            return 0
        else:
            return dot_product / (query_norm * document_norm)


if __name__ == "__main__":
    queries_frequencies = load_frequencies("queries", True)
    ir_system = IRSystem2(load_frequencies("documents"))
    for qu_id, query in enumerate(queries_frequencies[:10], start=1):
        query_vector = ir_system.vectoricer_query(query)
        similarities = []
        for doc_id, document_vector in enumerate(ir_system.documents_vectors, start=1):
            similarity = ir_system.cosine_similarity(query_vector, document_vector)
            similarities.append((doc_id, similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        print(qu_id)
        for doc_id, similarity in similarities[:10]:
            print(f"D{doc_id}: {similarity:.2f}")
