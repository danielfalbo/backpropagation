import math

class Value:

    def __init__(self, data, _parents=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_parents)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            # derivative of tan is 1 - tan^2
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(self):
            if self not in visited:
                visited.add(self)
                for c in self._prev:
                    build_topo(c)
                topo.append(self)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

if __name__ == '__main__':
    import torch

    _x1 = torch.Tensor([2.0])   ; _x1.requires_grad = True
    _x2 = torch.Tensor([0.0])   ; _x2.requires_grad = True

    _w1 = torch.Tensor([-3.0])  ; _w1.requires_grad = True
    _w2 = torch.Tensor([1.0])   ; _w2.requires_grad = True

    _b = torch.Tensor([6.88137]); _b.required_grad = True

    _n = _x1*_w1 + _x2*_w2 + _b
    _o = torch.tanh(_n)

    _o.backward()

    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')

    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')

    b = Value(6.88137, label='b')

    x1w1 = x1*w1; x1w1.label='x1w1'
    x2w2 = x2*w2; x2w2.label='x2w2'

    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label='x1w1 + x2w2'

    n = x1w1x2w2 + b; n.label='n'

    o = n.tanh(); o.label='o'

    REL_TOL = 1e-06

    assert(math.isclose(o.data, _o.data.item(), rel_tol=REL_TOL))

    o.backward()

    assert(math.isclose(x2.grad, _x2.grad.item(), rel_tol=REL_TOL))
    assert(math.isclose(w2.grad, _w2.grad.item(), rel_tol=REL_TOL))
    assert(math.isclose(x1.grad, _x1.grad.item(), rel_tol=REL_TOL))
    assert(math.isclose(w1.grad, _w1.grad.item(), rel_tol=REL_TOL))
    print('all tests successful')
