# Instructor Notes

## Mon Nov 25 (AM)

```python
print("Let's take a look!")
```

```python
def moveone()
    resp = send(Request.StartMove())
    ...

def turnone():
    ...

while True:
    resp = send(Request.ExitSensor())
    if isinstance(resp, Response.Exit())
        break
```

```python
def g():
    yield
```

```python
# function (“subroutine”)
def f():
    return
# inputs → [@] → outputs

# generator
def g():
    yield
# inputs → [@] → (partial) output
#              → (partial) output
#              → (partial) output
#              → (partial) output
#              → (partial) output

# generator (“coroutine”)
def g():
    _ = yield
# (partial) input → [@] → (partial) output
# (partial) input → [@] → (partial) output
# (partial) input → [@] → (partial) output
```

```python
def f(data):
    x = data + 1
    y = data * 2
    z = data ** 3
    return x, y, z

# from dis import dis
# dis(f)
print(f'{f.__code__.co_code = }')
```

```python
def g(data):
    x = data + 1
    yield x
    y = data * 2
    yield y
    z = data ** 3
    yield z

# from dis import dis
# dis(g)
print(f'{g.__code__.co_code = }')
```

```python
from inspect import currentframe, getouterframes

# class T:
#     def __del__(self):
#         print(f'T.__del__({self!r})')

def f(x):
    return g(x)

def g(x):
    return h(x)

def h(x):
    # xyz = T()
    return x ** 3, currentframe()

from gc import collect
collect()

rv = f(123)
# print(f'{rv = }')
print(f'{rv[-1].f_locals["xyz"] = :}')
# print(f'{signature(f) = }')
```

```python
def g(data):
    x = data + 1
    yield x
    y = data * 2
    yield y
    z = data ** 3
    yield z

gi = g(123)
print(f'{gi.gi_frame.f_locals = }')
print(f'{next(gi) = }')
print(f'{gi.gi_frame.f_locals = }')
print(f'{next(gi) = }')
print(f'{gi.gi_frame.f_locals = }')
print(f'{next(gi) = }')
print(f'{gi.gi_frame.f_locals = }')
```

Generators/Generator Coroutines at their core are way to decompose a large
computation into smaller parts (and to accomplish some goal from this
decomposition.)
- asyncio
- API simplification (start/stop modalities)
- alternate modeling/formalisation for state machines (← Bluesky)

```python
def newton(f, f_prime, x0):
    x = x0
    for _ in range(50):
        x -= f(x) / f_prime(x)
    return x

f       = lambda x: (x - 3) * (x + 4) # x² + x - 12
f_prime = lambda x: 2*x + 1

# roots:
#  x =  3
#  x = -4
print(f'{newton(f, f_prime, +5) = :>5}')
print(f'{newton(f, f_prime, -5) = :>5}')
```

```python
def newton(f, f_prime, x0, num_steps=50):
    x = x0
    for _ in range(num_steps):
        x -= f(x) / f_prime(x)
    return x

def newton(f, f_prime, x0, num_steps=50, abs_tol=None):
    x = x0
    for _ in range(num_steps):
        x -= f(x) / f_prime(x)
    return x

def newton(f, f_prime, x0, num_steps=50, abs_tol=None, rel_tol=None):
    x = x0
    for _ in range(num_steps):
        x -= f(x) / f_prime(x)
    return x

def newton(f, f_prime, x0, num_steps=50, abs_tol=None, rel_tol=None, total_time=None):
    x = x0
    for _ in range(num_steps):
        x -= f(x) / f_prime(x)
    return x

# Newton
# - step 0
#
# - step 1
#
# - step 2
#
# - ...
#
# - step N

def newton(f, f_prime, x0):
    x = x0
    while True:
        x -= f(x) / f_prime(x)
        yield x

f       = lambda x: (x - 3) * (x + 4) # x² + x - 12
f_prime = lambda x: 2*x + 1

from itertools import islice
from collections import deque

print(f'{deque(islice(newton(f, f_prime, 0), 100), maxlen=1)[0] = }')
```

```python
from scipy.optimize import newton
help(newton)
```
