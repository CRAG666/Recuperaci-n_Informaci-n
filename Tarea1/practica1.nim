import strutils, sequtils, tables

const ALPHABET: seq[char] = toSeq('A'..'Z')

proc loadStopwords(stopwordsFile: string): seq[string] =
  var stopwords: seq[string] = @[]
  for word in open(stopwordsFile).lines:
    let w = word.strip()
    if w notin ALPHABET:
      stopwords.add(w.toLower())
  result = stopwords

import tables

type
  Document = object
    Id: int
    Id_Psical: string
    Date: string
    Page: string
    Content: string

proc getMetadata(title: string): tuple[id: string, date: string, page: string] =
  let pageIndex = title.find("PAGE")
  let page = title.slice(pageIndex).split()
  let idDate = title[0..<pageIndex].split(maxSplits = 1)
  result = (idDate[0], idDate[1].rstrip(), page[1])

proc loadDocuments(documentsPath: string): seq[Document] =
  let text = readFile(documentsPath)
  var documents: seq[Document] = @[]

  for (id, fragment) in text.split("*TEXT")[1..^0].enumerate(1):
    let lines = fragment.strip().split("\n")

    if not lines.isNil:
      let (idPsical, date, page) = getMetadata(lines[0])
      let content = lines[1..^0].join(" ")
      let document = Document(id, idPsical, date, page, content)
      documents.add(document)

  result = documents

proc loadQueries(queriesPath: string): seq[string] =
  let text = readFile(queriesPath)
  var queries: seq[string] = @[]

  for (_, fragment) in text.split("*FIND")[1..^0].enumerate(1):
    let lines = fragment.strip().split("\n", maxSplits = 1)
    if not lines.isNil:
      queries.add(lines[1])

  result = queries
