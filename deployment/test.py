import pickle
import numpy as np 
import sklearn

with open("model.pkl", "rb") as f:
    model = pickle.load(f)
    
with open("normalizer.pkl", "rb") as g:
    normalizer = pickle.load(g) 
    
sample = np.array([1, 0, np.log(22)+1, 1, np.log(150773)+1, 0, 2])
sample = normalizer.transform([sample])

p = int(model.pdf(sample))

print(sample)