from graphviz import Digraph
import math
import random


def trace(root):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges


def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        dot.node(name=uid, label="{%s | data %.4f | grad %.4f}" % (
            n.label, n.data, n.grad), shape="record")
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2))+n2._op)
    return dot


visited = set()
topo = []


def build_topo(v):
    if v not in visited:
        visited.add(v)

        for chield in v._prev:
            build_topo(chield)
        topo.append(v)


class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._backward = lambda: None
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other ** -1

    def __neg__(self):
        return self * -1
    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __pow__(self, other):
        assert isinstance(other, (int, float)
                          ), "only supporting int/float powers for now"
        out = Value(self.data**other, (self, ), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x)+1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        visited = set()
        topo = []

        def build_topo(v):
            if v not in visited:
                visited.add(v)

                for chield in v._prev:
                    build_topo(chield)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(2.1, label='c')
e = a*b
e.label = 'e'
d = e+c
d.label = 'd'
f = Value(-3.3, label='f')


L = d * f
L.label = 'L'

L.grad = 1.0

e = (2*L).exp()
o = (e-1) / (e + 1)

build_topo(o)


o.backward()

dot = draw_dot(o)
dot.render("graph",view=True)



class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

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
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        #return [p for neuron in self.neurons for p in neuron.parameters()]
        param = []
        for neuron in self.neurons:
            ps = neuron.parameters()
            param.extend(ps)
        return param


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
    
    
    
x = [
     1.0,
    -1.0,
     1.0,
     1.0,
    -1.0,
    -1.0,
     1.0,
    -1.0,
     1.0,
    -1.0
]


xs = [
    [0.2,  1.1, -0.5, 2.0, 0.3, -1.2, 0.7,  1.5],
    [1.4, -0.2,  0.8, 0.5, 1.7,  0.3, -0.4, 0.9],
    [-0.7, 0.4,  1.2, 1.8, -0.6, 0.5, 1.1, -0.3],
    [2.1,  1.3, -1.1, 0.2, 0.8, -0.9, 1.4,  0.6],
    [0.5, -1.5,  0.3, 1.1, 0.9,  1.2, -0.8, 0.4],
    [-1.2, 0.7,  1.5, 0.3, -0.4, 1.8, 0.2,  1.0],
    [1.0,  0.8, -0.2, 1.4, 0.6, -1.0, 0.5,  1.7],
    [-0.3, 1.6,  0.9, -0.7, 1.3, 0.4, 1.2, -0.5],
    [1.7, -0.8,  0.4, 1.9, -1.1, 0.7, 0.3,  1.2],
    [0.6,  1.2, -0.9, 0.8, 1.5, -0.3, 0.9,  0.1],
]

ys = [
     1.0,
    -1.0,
     1.0,
     1.0,
    -1.0,
    -1.0,
     1.0,
    -1.0,
     1.0,
    -1.0
]

n = MLP(8, [16, 16, 8, 1])

dot = draw_dot(n(x))
dot.render("graph",view=True)


for k in range(20):
    #forward pass
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys,ypred)) / len(xs)
    
    #backward pass
    
    
    for p in n.parameters():
        p.grad = 0.0
        
    loss.backward()
    
    for p in n.parameters():
        p.data += -0.01 * p.grad
    print(k,loss.data)