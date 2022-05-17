"""
This section covers utility functions for Named Entity 
Recognition models.
Author: Charangan Vasantharajan
"""
class SentenceGetter(object):
    """
    Args:
        data : dataframe
          data
        sentences : list
          list of sentences of the data in the form of tuple(word ,tag)

    Returns:
        Nothing
    """

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False

        def agg_func(s): return [
            (w, t) for w, t in zip(s["Word"].values.tolist(), s["Tag"].values.tolist())
        ]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None
