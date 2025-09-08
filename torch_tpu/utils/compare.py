import numpy as np


def cos_sim(vector_a, vector_b):
    vector_a = vector_a.reshape(-1)
    vector_b = vector_b.reshape(-1)
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    with np.errstate(invalid="ignore"):
        cos = np.nan_to_num(num / denom)
    sim = 0.5 + 0.5 * cos
    return sim
