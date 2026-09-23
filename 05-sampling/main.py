import numpy as np

logits = np.array([
    8.2,   # sunny
    7.4,   # rainy
    6.9,   # cloudy
    2.1,   # terrible
    0.3    # banana
])

def softmax(values):
    exp_values = np.exp(values - np.max(values))
    return exp_values / exp_values.sum()


for temperature in [0.2, 0.5, 1.0, 2.0]:
    scaled_logits = logits / temperature

    probabilites = softmax(scaled_logits)

    print("=" * 50)
    print("Temperature: ", temperature)

    for probability in probabilites:
        print(f"{probability:.4f}")

# Top-K = helps in restricting choice in selection after model generation

# Top-P = probability based candidate count