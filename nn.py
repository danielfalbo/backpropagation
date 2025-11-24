import random
from micrograd import Value

class Neuron:

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]

class Layer:

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

if __name__ == "__main__":
    import math

    n = MLP(3, [4, 4, 1])

    # inputs
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]

    # desired targets
    ys = list(map(Value, [1.0, -1.0, -1.0, 1.0]))

    EPOCHS = 2000
    STEP = 0.05

    for k in range(EPOCHS):
        # forward pass
        ypred = [n(map(Value, x))[0] for x in xs]

        # compute loss
        loss = sum([(yout - ygt)*(yout - ygt) for ygt, yout in zip(ys, ypred)],
                    Value(0.0))

        # backward pass
        for p in n.parameters():
            p.grad = 0.0
        loss.backward()

        # follow the gradient to decrease loss
        for p in n.parameters():
            p.data -= STEP * p.grad

        print(f"epoch: {k}, loss: {loss.data}")

    print()

    REL_TOL = 1e-01
    assert(all(math.isclose(ygt.data, yout.data, rel_tol=REL_TOL)
        for ygt, yout in zip(ys, ypred)))

    print("all tests successful")
