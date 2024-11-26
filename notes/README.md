# Instructor Notes

## Mon Nov 25

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

```python
def test():
    yield Request.Test()

def move_one():
    yield Request.Move()
    ... # ???
    yield Request.StopMove()
```

```python
def test():
    return Request.Test()

def move_one():
    return [Request.Move(), ..., Request.StopMove()]
```

```python
def test():
    yield Request.Test()

def move_one():
    yield Request.Move()
    ... # get information about whether you have moved
    yield Request.StopMove()
```

```python
from pathlib import Path
from subprocess import check_output
from shlex import quote
from collections import namedtuple

data_dir = Path('data')

Rename = namedtuple('Rename', 'src dst')
Remove = namedtuple('Remove', 'target')

checksums = {*()}
actions = []
for p in data_dir.iterdir():
    if p.is_file():
        result = check_output(['xxh64sum', quote(f'{p!s}')]).decode()
        if result in checksums:
            actions.append(Remove(p))
            # p.unlink()
        elif:
            ...
            actions.add(Rename(..., ...))
            ...
        else:
            checksums.add(result)

for act in actions:
    act()
```

```python
def f():
    for x in range(10):
        print(f'{x = }')

f()
```

```python
def g():
    for x in range(10):
        print(f'{x = }')
        yield

# gi = iter(g())
# while True:
#     try:
#         next(gi)
#     except StopIteration:
#         break

for _ in g():
    pass
```

```python
def c(x):
    yield x +  1
    yield x *  2
    yield x ** 3

ci = c(123)
# print(f'{ci = }')
print(f'{next(ci) = :,}')
print(f'{next(ci) = :,}')
print(f'{next(ci) = :,}')
```

```python
def c(x):
    yield x + 1
    x = yield x * 2
    x = yield x ** 3

ci = c(123)
print(f'{next(ci)     = :,}') # 123 +  1 = 124             | (x := 123) +  1 = 124
print(f'{ci.send(456) = :,}') # 456 *  2 = 912             | (x := 123) *  2 = 246
print(f'{ci.send(789) = :,}') # 789 ** 3 = some big number | (x := 789) ** 3 = ...
```

```python
from asyncio import run, sleep as aio_sleep, gather

async def task(name):
    while True:
        print(f'task {name = }')
        await aio_sleep(1)

@lambda f: run(f())
async def main():
    await gather(task('task#0'), task('task#1'))
```

```python
#           stuck
#      /       \        \
#   (left    (right    (both
#    sensor   sensor    obs.)
#    unobs.)  unobs.)    |
#     |         |        |
#   turn      turn     turn
#   left      right    around
```

```python
from collections import deque
from dataclasses import dataclass

# even → left
# odd  → right
#
#         a
#        / \
#       b   c
#      / \   \
#     d   e   f (cannot visit >3; skip, otherwise)
#     |   |   | (skip, if we have visit b>2 in the last hour)
#     a   a   a'
#              \
#               a''
#                \
#                 a'''

@dataclass(frozen=True, unsafe_hash=True)
class State:
    name : str
    def __call__(self):
        print(f'{self!r}')

states = {} # nodes
for name in 'abcdef':
    states[name] = State(name)

transitions = { # edges
    states['a']: {
        states['b']: (lambda inp: inp % 2 == 0, None),
        states['c']: (lambda inp: inp % 2 == 1, None),
    },
    states['b']: {
        states['d']: (lambda inp: inp % 2 == 0, None),
        states['e']: (lambda inp: inp % 2 == 1, None),
    },
    states['c']: {states['f']: (lambda _: True, 3)},
    states['d']: {states['a']: (lambda _: True, None)},
    states['e']: {states['a']: (lambda _: True, None)},
    states['f']: {states['a']: (lambda _: True, None)},
}

from collections import Counter
def executor(states, transitions, initial_state, inputs):
    state = initial_state
    node_visits = Counter()
    while inputs:
        node_visits[state] += 1
        state()
        inp = inputs.popleft()
        for next_state, (pred, max_node_visits) in transitions[state].items():
            if pred(inp) and (max_node_visits is not None and node_visits[next_state] < max_node_visits):
                state = next_state
    state()

if __name__ == '__main__':
    from random import Random
    rnd = Random(0)
    inputs = deque([rnd.randint(-10, +10) for _ in range(10)])
    executor(states, transitions, states['a'], inputs)
    # print(f'{inputs = }')
    #   [2, 3, -9, -2, 6, 5, 2, -1, 5, 1]
    #  a b  e   a   b  d  a  b   e  a  c
```

```python
#   a
#  / \
# b   c
```

```python
from collections import deque
from dataclasses import dataclass

# even → left
# odd  → right
#
#         a
#        / \
#       b   c
#      / \   \
#     d   e   f (cannot visit >3; skip, otherwise)
#     |   |   | (skip, if we have visit b>2 in the last hour)
#     a   a   a

@dataclass(frozen=True, unsafe_hash=True)
class State:
    name : str
    def __call__(self):
        print(f'{self!r}')

states = {} # nodes
for name in 'abcdef':
    states[name] = State(name)

def machine():
    f_visits = 0
    b_visits = 0
    while True:
        inp = yield states['a']
        if inp % 2 == 0:
            b_visits += 1
            inp = yield state['b']
            if inp % 2 == 0:
                _ = yield state['d']
            else:
                _ = yield state['e']
        else:
            _ = yield state['c']
            f_visits += 1
            if f_visits < 3 or b_visits < 2:
                _ = yield state['f']


if __name__ == '__main__':
    from random import Random
    rnd = Random(0)
    inputs = deque([rnd.randint(-10, +10) for _ in range(10)])

    m = machine(inputs)
    state = next(m)
    for inp in inputs:
        state()
        state = m.send(inp)
```

## Tue Nov 26

```python
def move():
    yield Start
    while True:
        resp = yield Check
    yield Stop

def move():
    yield send(Start)
    while True:
        yield send(Check)

def executor():
    m = move()
    while True:
        if next(m):
            break
```

```python
from requests import get as req_get

def get(*args, **kwargs):
    for _ in range(2):
        try:
            return req_get(*args, **kwargs)
        except HttpError:
            pass

def f():
    if get('http://api/endpoint').json()['result']:
        return g(), h()
    else:
        return ...

def g():
    return get('http://api/other-endpoint')

def h():
    return get('http://api/one-more-endpoint')

if __main__ == '__name__':
    f()
```

```python
from requests import get

def f():
    if (res := yield 'http://api/endpoint'):
        yield from g()
        yield from h()
    else:
        ...

def g():
    yield 'http://api/other-endpoint'

def h():
    yield 'http://api/one-more-endpoint'

if __name__ == '__main__':
    inst = f()
    res = None
    retry_counts = 0
    while True:
        try:
            url = inst.send(res)
            try:
                res = get(url).json()['result']
            except RandomApiError:
                retry_counts += 1
                if retry_counts > 10:
                    raise
        except StopIteration:
            break
```

```python
def state_machine(send):
    while True:
        if send(Response.CheckExit()):
            pass
```

```python
def coro():
    _ = yield 123

from dis import dis
dis(coro)
# ci = coro()
# ci.send(123)
```

```python
from itertools import chain

def g0():
    yield 1
    yield 2
    yield 3

def g1():
    yield 4
    yield 5
    yield 6

for x in chain(g0(), g1()):
    print(f'{x = }')
```

```python
def coro(x=None):
    while True:
        x = yield x

ci = coro(123)
next(ci) # first step
while True: # other step
    ci.send(...)
```

```python
def coro0():
    ...
def coro1():
    ...

# coro0
# first - rest - ... - rest

# coro1
# first - rest - ... - rest

# chain(coro0, coro1)
# first - rest - ... - rest - first - rest - ... - rest
#  x       .      .     .      x        .     .     .
```

```python
def repeat(value):
    while True:
        yield value

for x in repeat(123):
    print(f'{x = }')
```

```python
def latch(value):
    while True:
        new_value = yield value
        if new_value is not None:
            value = new_value

ci = latch(123); next(ci)
# print(f'{next(ci)     = }')
print(f'{ci.send(456) = }')
print(f'{next(ci)     = }')
print(f'{next(ci)     = }')
print(f'{next(ci)     = }')
```

```python
from functools import wraps

def pumped(coro):
    @wraps(coro)
    def inner(*args, **kwargs):
        ci = coro(*args, **kwargs)
        next(ci)
        return ci
    return inner

@pumped
def latch():
    value = yield
    while True:
        new_value = yield value
        if new_value is not None:
            value = new_value

@pumped
def latch(value):
    while True:
        new_value = yield value
        if new_value is not None:
            value = new_value

ci = latch()
# print(f'{next(ci)     = }')
print(f'{ci.send(456) = }')
print(f'{next(ci)     = }')
print(f'{next(ci)     = }')
print(f'{next(ci)     = }')
```

```python
# first → rest → ...  → rest
# rest  → ...  → rest → last
```

```python
from functools import wraps

def pumped(coro):
    @wraps(coro)
    def inner(*args, **kwargs):
        ci = coro(*args, **kwargs)
        next(ci)
        return ci
    return inner

# @pumped
def coro():
    # print(f'{(yield)     = }')
    print(f'{(yield 123) = }')
    print(f'{(yield 456) = }')
    print(f'{(yield 789) = }')

ci = coro()
print(f'{next(ci)     = }')
print(f'{ci.send(...) = }')
print(f'{ci.send(...) = }')
print(f'{ci.send(...) = }')
# print(f'{ci.send(...) = }')
```

```python
def coro():
    yield ...
    yield ...
    yield ...
    return ...
```

```python
def f(data, *, mode):
    if mode:
        return ...
    else:
        return ...

def g(data, *, mode):
    while True:
        if mode:
            yield ...
        else:
            yield ...

@pumped
def coro(*, mode):
    while True:
        if mode:
            data = yield ...
        else:
            data = yield ...
```

Should everything be pumped?

```python
from functools import wraps

def pumped(coro):
    @wraps(coro)
    def inner(*args, **kwargs):
        ci = coro(*args, **kwargs)
        next(ci)
        return ci
    return inner

@pumped
def coro():
    print(f'{(yield)      = }')
    print(f'{(yield  123) = }')
    print(f'{(yield  456) = }')
    print(f'{(yield  789) = }')
    return 999

ci = coro()
next(ci)
for _ in range(3):
    print(f'{ci.send(...) = }')
try:
    next(ci)
except StopIteration as e:
    print(f'{e = }')
```

### Callum & Abby

- Roman & Kari
- Nate & Seher
- Eugene & Eric

message payload.
```python
```

### Shekar & Mark

- Abigail & Jun
- Robert & Juan

send_command

```python
```

### Amendments

Adjusting the Request Stream:
- Relative Moves/Synthetic Moves
- Fusing/Splitting/Replacing Moves
- State Reconstruction
