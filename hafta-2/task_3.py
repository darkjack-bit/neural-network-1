import math


class Value:
    def __init__(
        self,
        data,
        _children=(),
        _op="",
        label=""
    ):
        self.data = data
        self.grad = 0.0

        self._prev = set(_children)
        self._backward = lambda: None

        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = (
            other
            if isinstance(other, Value)
            else Value(other)
        )

        out = Value(
            self.data + other.data,
            (self, other),
            "+"
        )

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        other = (
            other
            if isinstance(other, Value)
            else Value(other)
        )

        out = Value(
            self.data * other.data,
            (self, other),
            "*"
        )

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def tanh(self):
        x = self.data

        t = (
            math.exp(2 * x) - 1
        ) / (
            math.exp(2 * x) + 1
        )

        out = Value(
            t,
            (self,),
            "tanh"
        )

        def _backward():
            self.grad += (
                (1 - t**2)
                * out.grad
            )

        out._backward = _backward

        return out

    def backward(self):
        visited = set()
        topo = []

        def build_topo(v):
            if v not in visited:
                visited.add(v)

                for child in v._prev:
                    build_topo(child)

                topo.append(v)

        build_topo(self)

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()
            
a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = Value(2.1, label="c")

e = a * b
e.label = "e"

d = e + c
d.label = "d"

f = Value(-3.3, label="f")

L = d * f
L.label = "L"

L.backward()


print("a:", a.grad)
print("b:", b.grad)
print("c:", c.grad)
print("d:", d.grad)
print("e:", e.grad)
print("f:", f.grad)
print("L:", L.grad)