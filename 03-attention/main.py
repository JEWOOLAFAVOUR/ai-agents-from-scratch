import numpy as np

query = np.array([1.0, 0.0])

keys = np.array([
    [1.0, 0.0],   # animal
    [0.0, 1.0],   # road
    [0.8, 0.1],   # tired
])

scores = keys @ query

print("Attention scores: ", scores)