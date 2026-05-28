def stream_data(chain, query):
    for i in chain.stream({'input': query}):
        if 'answer' in i:
            yield i['answer']