import mlx.core as mx


X = mx.random.normal((100, 1))
y = 2 * X + 1


# Parameters
W = mx.random.normal((1, 1))
b = mx.zeros((1,))


# Guesser
def model(X, W, b):
    return X @ W + b


# loss: To measure how wrong the model's guess is by comparing it to the correct answer y.
def loss_fn(W, b, X, y):
    predictions = model(X, W, b)
    return mx.mean(mx.square(predictions - y))


# gradient: direction
grad_fn = mx.grad(loss_fn, argnums=(0, 1))

# step size
learning_rate = 0.1

# loop(training)
for i in range(20):
    # direction to improve our model guess model()
    grads_W, grads_b = grad_fn(W, b, X, y)
    W = W - learning_rate * grads_W
    b = b - learning_rate * grads_b
    mx.eval(W, b)

    loss = loss_fn(W, b, X, y)
    print("Step:", i + 1, "Loss:", loss.item())

print("Training done")
print("Learned W:", W.item(), "Learned b:", b.item())
