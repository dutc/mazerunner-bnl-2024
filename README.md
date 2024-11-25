# µBluesky (“Maze Runner”)

## Overview

You have a robot that needs to navigate a (small, square) maze (with 90° walls
and rigid, unit spacing.) There is no global information about the layout of
the maze.

## Details

The robot can move forward but cannot move backward. The robot can turn in
precise 90° increments clockwise or anticlockwise. You can communicate with the
robot via the following request messages:

- start or stop moving (forward-only; snapped to single spaces)
- start or stop turning clockwise or anticlockwise (snapped to 90° angles)
- read front/left/right bumper sensor (i.e., has it hit a wall)
- read overhead sensor (i.e., has it found the exit)

For each of the above request messages, the robot will send a response to indicate its state:

- start moving → robot has started moving
- stop moving → robot has stopped moving & how far it has moved since it was last instructed (snapped to single spaces)
- start turning → robot has started turning
- stop turning → robot has stopped turning & how far it has turned since it was last instructed (snapped to 90° angles)
- read bumper → whether the robot is up against a wall
- read overhead → whether the robot has exited the maze

The robot is guaranteed to move in exact single spaces and turn at exact 90°
angles; however, the robot's motors will have arbitrary delays before starting
an action, and movement or turning will sometimes be slower or faster.
Therefore, you must confirm the action via the response.

**Goal**: navigate the robot to the exit.

## Questions & Tasks

Within the above loose structure, we will seek to answer the following
questions and complete the following tasks, interleaving general discussion,
instruction on theory, and independent working time on tasks.

1. Navigate the robot through a linear maze (one single corridor)
2. Navigate the robot through an elbow (one single corridor with a left- or right-turn)
3. Navigate the robot through a T-junction where one side is a dead-end & the other is the exit
4. Navigate the robot through an arbitrary maze (with left-wall hugging)
5. (Optional) Accommodate messaging faults (arbitrarily delayed or lost messages)
6. (Optional) Navigate multiple robots through a single maze with multiple entrances (where robots can impede each other's progress)

- What is a generator in Python?
- What is the difference between `yield` and `return`?
- What is PEP-380 delegation to a subgenerator and `yield from`?
- What is a generator coroutine in Python?
- How do generator coroutine modelings of state machines differ from traditional states/transition modelings?
- What are immediate vs retained modes of operation?
- What does `return` in a generator coroutine do?
- What is the `itertools` module?
- What are iteration helpers? How and why do I want to write my own? How do I write these properly?
- What is pumping or priming? What are first vs last modalities?
