from gensim.corpora import Dictionary
from gensim.models import LdaModel


class LDA:
    def __init__(self, docs, num_topics=10, passes=20, iterations=400) -> None:
        processed_docs = [doc.lower().split() for doc in docs if doc]

        if not processed_docs:
            raise ValueError("No documents provided for LDA training.")

        self.dictionary = Dictionary(processed_docs)
        self.dictionary.filter_extremes(no_below=2, no_above=0.9)

        self.corpus = [self.dictionary.doc2bow(doc) for doc in processed_docs]

        self.lda_model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=num_topics,
            passes=passes,
            iterations=iterations,
            alpha='auto',
            eta='auto'
        )
