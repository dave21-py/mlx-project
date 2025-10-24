import mlx.core as mx


a = mx.array([1, 2, 3])
b = mx.array([4, 5, 6])


c = a * b + 10

print("a:", a)
print("b:", b)
print("c:", c)
print("Device used:", mx.default_device())
