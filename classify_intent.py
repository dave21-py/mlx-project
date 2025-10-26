from multiprocessing import Lock
import mlx.core as mx
import mlx.nn as nn

sentences = [
    # Timer[0]
    "set a timer for 10 minutes",
    "remind me in 5 minutes",
    "start a countdown",
    # Music[1]
    "play some rock music",
    "turn on the radio",
    "i want to hear a song",
    # Weather[2]
    "will it be sunny tomorrow",
    "what is the weather today",
    "do i need an umbrella",
]

labels = mx.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

# vocabulary
all_words = set()
for sentence in sentences:
    words = sentence.split()
    for word in words:
        all_words.add(word)

vocabulary = sorted(list(all_words))

# vectorization
X = []
for sentence in sentences:
    sentence_vector = [0] * len(vocabulary)
    words_in_sentence = sentence.split()
    for word in words_in_sentence:
        if word in vocabulary:
            index = vocabulary.index(word)  # index of timer maybe 28
            sentence_vector[index] = 1  # word found
    X.append(sentence_vector)


X = mx.array(X)

# print("\nFirst sentence:", sentences[0])
# print("\nIts vector:", X[0])

# print("\nSecond sentence:", sentences[1])
# print("\nIts vector:", X[1])


# Model setup
num_words = len(vocabulary)
num_classes = 3

W = mx.random.normal((num_words, num_classes))  # [i,j]
b = mx.zeros(num_classes)

# print("Shape of W:", W.shape)
# print("Shape of b:", b.shape)


def model(X, W, b):
    return X @ W + b


test_logits = model(X[0], W, b)
# print("\n--- Model Test ---")
# print("First sentence vector shape:", X[0].shape)
# print("Untrained model's raw scores for the first sentence:", test_logits)


# H Loss- W, L Loss- R
def loss_fn(W, b, X, labels):
    logits = model(X, W, b)
    return mx.mean(
        nn.losses.cross_entropy(logits, labels)
    )  # model scores and actual correct scores


grad_fn = mx.grad(loss_fn, argnums=(0, 1))

learning_rate = 0.1
for i in range(50):
    grads_W, grads_b = grad_fn(W, b, X, labels)
    W = W - learning_rate * grads_W
    b = b - learning_rate * grads_b
    mx.eval(W, b)
    loss = loss_fn(W, b, X, labels)
    # print("Step:", i + 1, "Loss:", loss.item())


final_scores = model(X, W, b)
predicted_labels = mx.argmax(final_scores, axis=1)

print("\n--- Model Predictions ---")
print("Correct labels:   ", labels)
print("Predicted labels: ", predicted_labels)
