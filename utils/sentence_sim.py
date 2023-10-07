import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class Sentence_sim(object):

    def __init__(self):
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        self.bert = hub.load(module_url)

    def cal_sentence_sim(self, s1, s2):
        message = [s1, s2]
        message_embedding = self.bert(message)
        sentence_similarity = (np.around(cosine_similarity([message_embedding[0]], message_embedding[1:]).tolist()[0],
                                decimals=5))[0]
        return sentence_similarity


#word = "I am a boy."
#sentence = "I am a bird."

#met = Sentence_sim()
#print(met.cal_sentence_sim(word, sentence))
