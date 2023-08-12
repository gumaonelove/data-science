from flask import Flask, render_template, request
from time import time
from timer import Timer
from collections import defaultdict
from data import get_data, Document

import numpy as np
from gensim.models import Word2Vec


data = get_data(low=False)
t = Timer()
app = Flask(__name__, template_folder='.')

index = defaultdict(list)
documents = {}

# Расстояние между предложений с помощью world2vec
w2v_model = Word2Vec(
    min_count=10,
    window=2,
    negative=10,
    alpha=0.03,
    min_alpha=0.0007,
    sample=6e-5,
    sg=1
)

t.start('Начали обучение Word2Vec')
w2v_model.build_vocab(data[:, 0])

w2v_model.train(
    data[:, 0],
    total_examples=w2v_model.corpus_count,
    epochs=30,
    report_delay=1
)

w2v_model.init_sims(replace=True)
t.stop("Обучение Word2Vec заняло ")


def most_similar(word: str) -> list:
    return w2v_model.wv.most_similar(positive=[word])


def similarity( word1: str, word2: str):
    return w2v_model.wv.similarity(word1, word2)


def distanse_matrix(title_1: list, title_2: list):
    m, n = len(title_1), len(title_2)
    matrix = [[0]*n for _ in range(m)]

    for i in range(m):
        for j in range(n):
            try:
                matrix[i][j] = similarity(title_1[i], title_2[j])
                print(similarity(title_1[i], title_2[i]))
            except KeyError:
                pass
    return np.array(matrix).mean()


def build_index():
    '''Считывает сырые данные и строит индекс'''
    t.start('Начали строить индекс')
    for i, arr in enumerate(data):
        title, authors = arr
        for word in title.split():
            index[word.lower()].append(i)
        for author in authors.split(','):
            index[author.lower()].append(i)
        documents[i] = Document(*arr)
    t.stop("Построение индекса заняло ")


def score(query, document):
    scor = distanse_matrix(query.split(), document.title.split())
    return scor


def retrieve(query, limit: int = 10) -> list:
    '''Возвращает начальный список релевантных документов'''
    candidates = []
    count = 0

    for word in query.split():
        for candidate_id in index[word]:
            candidates.append(documents[candidate_id])
        count += len(index[word])
        if count >= limit: break
    return candidates[: limit]

build_index()

@app.route('/', methods=['GET'])
def index_method():
    start_time = time()
    query = request.args.get('query')
    if query is None:
        query = ''
    documents = retrieve(query)
    scored = [(doc, score(query, doc)) for doc in documents]
    scored = sorted(scored, key=lambda doc: -doc[1])
    results = [doc.format(query)+['%.2f' % scr] for doc, scr in scored]

    return render_template(
        'index.html',
        time="%.2f" % (time()-start_time),
        query=query,
        search_engine_name='Tinkoff',
        results=results
    )


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8080, use_reloader=False)