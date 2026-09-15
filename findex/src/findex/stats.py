import tracemalloc
import time
from collections import Counter
from src.findex.core import iter_documents, tokenize

def measure_lazy(corpus_path):
    tracemalloc.start()
    start_time = time.time()
    
    vocab = Counter()
    
    for doc in iter_documents(corpus_path):
        for token in tokenize(doc):
            vocab[token] += 1
            
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return time.time() - start_time, peak_memory / (1024 * 1024), vocab.most_common(5)

def measure_greedy(corpus_path):
    tracemalloc.start()
    start_time = time.time()
    
    # Жадібний метод: завантажує ВСЕ в оперативну пам'ять
    docs = list(iter_documents(corpus_path)) 
    tokens = [token for doc in docs for token in tokenize(doc)]
    vocab = Counter(tokens)
    
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return time.time() - start_time, peak_memory / (1024 * 1024)