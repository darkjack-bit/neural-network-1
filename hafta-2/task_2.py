from graphviz import Digraph
import math

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
    dot = Digraph(
        format="svg",
        graph_attr={"rankdir": "LR"}
    )

    nodes, edges = trace(root)

    for n in nodes:
        uid = str(id(n))

        dot.node(
            name=uid,
            label="{%s | data %.4f | grad %.4f}" % (
                n.label,
                n.data,
                n.grad
            ),
            shape="record"
        )

        if n._op:
            dot.node(
                name=uid + n._op,
                label=n._op
            )

            dot.edge(
                uid + n._op,
                uid
            )

    for n1, n2 in edges:
        dot.edge(
            str(id(n1)),
            str(id(n2)) + n2._op
        )

    return dot


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
        self._op = _op
        self.label = label

        self._backward = lambda: None

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

        return out




a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = Value(2.1, label="c")

e = a * b
e.label = "e"

d = e + c
d.label = "d"

f = Value(-3.3, label="f")

L = d * f

"""
dL/dd = L(d,f) = d*f  = L(d+h,f) = (d+h)*f = ((d+h)*f - d*f)/h = f = (f*d + f*h - d*f) /h = f*h/h = f = -3.3

-3.3*1 = dl/de


D(e,c) = e+c = L(e+h,c) = e+h+c = (e+h+c - (e+c))/h = (e+h+c -e -c) / h = 1

"""



L.label = "L"

L.grad = 1.0
d.grad = -3.3

dot = draw_dot(L)
dot.render("graph",view=True)

x1 = Value(2.0, label="x1")
x2 = Value(0.0, label="x2")

w1 = Value(-3.0, label="w1")
w2 = Value(1.0, label="w2")

b = Value(6.881373587, label="b")

x1w1 = x1 * w1
x1w1.label = "x1*w1"

x2w2 = x2 * w2
x2w2.label = "x2*w2"

x1w1x2w2 = x1w1 + x2w2
x1w1x2w2.label = "x1*w1 + x2*w2"

n = x1w1x2w2 + b
n.label = "n"

o = n.tanh()
o.label = "o"

dot = draw_dot(o)
dot.render("graph",view=True)