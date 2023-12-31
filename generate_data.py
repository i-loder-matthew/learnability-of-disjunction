import numpy as np
import pandas as pd
import pickle
import language as lang

# TODO Clean this script up

def generate_model(size = 20):

    info_request = np.full(size, np.random.choice(2, 1, p=[0.75, 0.25]))
    
    model = np.zeros((size, 5))

    model[:,0] = info_request

    weights = np.array([0.7, 0.1, 0.1, 0.1])

    np.random.shuffle(weights)

    J = np.random.choice(4, size, p=weights)
    model[np.arange(size), J + 1] = 1

    return model


models = []
for i in range(100000):
    models.append(generate_model())


english = lang.Language(1, 2, 2, 1, 0)

strings = []
for m in models:
    strings.append(english.generate_str_from_model(m))

data_english = pd.DataFrame([models, strings]).transpose()
data_english.columns = ['src', 'trg']
pd.to_pickle(data_english, "./data/data_english")


