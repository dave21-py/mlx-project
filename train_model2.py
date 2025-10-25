import mlx.core as mx


X = mx.random.normal((100, 1))
y = -3 * X + 5  # Inital

W = mx.random.normal((1, 1))
b = mx.zeros((1,))


def model(X, W, b):
    return X @ W + b


def loss_fn(W, b, X, y):
    predictions = model(X, W, b)
    return mx.mean(mx.square(predictions - y))


grad_fn = mx.grad(loss_fn, argnums=(0, 1))

learning_rate = 0.1
for i in range(20):
    grads_W, grads_b = grad_fn(W, b, X, y)
    W = W - learning_rate * grads_W
    b = b - learning_rate * grads_b
    mx.eval(W, b)
    loss = loss_fn(W, b, X, y)
    print("Step:", i + 1, "Loss:", loss.item())

print("TC")
print("Predicted W:", W.item(), "Predicted b:", b.item())
