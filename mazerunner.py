#!/usr/bin/env python3

from argparse import ArgumentParser
from asyncio import run, start_server, TaskGroup, sleep as aio_sleep
from atexit import register as atexit_register
from collections import namedtuple
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from itertools import cycle, chain, repeat
from functools import cached_property, wraps
from logging import getLogger, basicConfig, DEBUG, INFO
from multiprocessing import Process
from os import kill
from pathlib import Path
from pickle import loads, dumps
from random import Random
from signal import SIGTERM
from socket import socket, AF_INET, SOCK_STREAM
from sys import exit
from textwrap import dedent, indent
from time import sleep
from typing import Generator
from types import SimpleNamespace

from numpy import array, full, ndarray

logger = getLogger(__name__)

def bidicycle(xs):
    xs = [*xs]
    idx = 0

    def cur():
        while True:
            yield xs[idx]

    def fwd():
        nonlocal idx
        while True:
            idx = (idx + 1) % len(xs)
            yield xs[idx]

    def rev():
        nonlocal idx
        while True:
            idx = (idx - 1) % len(xs)
            yield xs[idx]

    return cur(), fwd(), rev()

class Message:
    def serialize(self):
        return dumps(self)
    @classmethod
    def deserialize(cls, payload):
        return loads(payload)
    def __init_subclass__(cls):
        setattr(Message, cls.__name__, cls)
class Request(Message):
    def __init_subclass__(cls):
        setattr(Request, cls.__name__, cls)
        super().__init_subclass__()
class Response(Message):
    def __init_subclass__(cls):
        setattr(Request, cls.__name__, cls)
        super().__init_subclass__()
for msg in dedent('''
    PowerOn PowerOff
    Move StopMove CheckMove
    TurnLeft TurnRight StopTurn CheckTurn
    FrontSensor LeftSensor RightSensor ExitSensor Test
''').strip().split():
    globals()[msg] = dataclass(frozen=True)(type(msg, (Request,), {}))
for msg in dedent('''
    PoweredOn PoweredOff
    MovingStart MovingStop
    TurningStart TurningStop
    Wall NoWall
    Exit NoExit
    TestSuccess
    Error
''').strip().split():
    globals()[msg] = dataclass(frozen=True)(type(msg, (Response,), {}))
@dataclass
class TurningState(Response):
    turns : int
@dataclass
class MovingState(Response):
    distance : int

class Tile(Enum):
    Wall, Floor, Exit  = 'x', '.', '*'
    @classmethod
    def from_symbol(cls, sym):
        return cls.symbols[sym]
Tile.symbols = {x.value: x for x in Tile.__members__.values()}

@dataclass
class Maze:
    start : tuple[int, int]
    maze  : ndarray

    @classmethod
    def from_file(cls, filename):
        with open(filename) as f:
            lines = (
                (ln[:ln.index('#')] if '#' in ln else ln)
                    .rstrip('\n')
                    [::2]
                for ln in f
            )
            lines = [*filter(None, lines)]

        maze = full(
            fill_value=Tile.Floor,
            shape=(len(lines), len(lines[0])),
        )

        for rownum, row in enumerate(lines):
            for colnum, col in enumerate(row):
                if col == '@':
                    start = rownum, colnum
                    tile = Tile.Floor
                elif (tile := Tile.from_symbol(col)) is not Tile.Floor:
                    maze[rownum, colnum] = tile

        return cls(
            maze=maze,
            start=start,
        )

def overlay(lines, symbols : dict[tuple[int, int], str]):
    for rownum, row in enumerate(lines):
        yield [
            symbols.get((rownum, colnum), col)
            for colnum, col in enumerate(row)
        ]

def render_maze(maze : Maze | ndarray):
    maze = maze.maze if isinstance(maze, Maze) else maze
    for row in maze:
        yield [tile.value for tile in row]

def pumped(coro):
    @wraps(coro)
    def inner(*args, **kwargs):
        ci = coro(*args, **kwargs)
        next(ci)
        return ci
    return inner

def agent_process(*, host, port, maze, seed, errors, tick, power_cycle_freq):
    rnd = Random(seed)

    maze = Maze.from_file(maze)
    for row in overlay(render_maze(maze), {maze.start: '@'}):
        logger.debug('Maze: %s', ' '.join(row))

    AgentPower = Enum('AgentPower', 'On Off')
    AgentState = Enum('AgentState', 'Moving TurningLeft TurningRight')

    Location = namedtuple('Location', 'angle forward left right')

    @dataclass
    class Agent:
        state : AgentState
        location : tuple[int, int]
        maze : Maze

        power_cycle : Generator
        power : AgentPower = AgentPower.Off

        tick : int = 1
        total_ticks : int = 0

        prev_location : tuple[int, int] = None
        num_turns : int = 0

        def __post_init__(self):
            self._hold, self._turn_right, self._turn_left = bidicycle([
                Location(
                    angle=0,
                    forward=lambda y, x: (y - 1, x    ),
                    left   =lambda y, x: (y    , x - 1),
                    right  =lambda y, x: (y    , x + 1),
                ),
                Location(
                    angle=90,
                    forward=lambda y, x: (y    , x + 1),
                    left   =lambda y, x: (y - 1, x    ),
                    right  =lambda y, x: (y + 1, x    ),
                ),
                Location(
                    angle=180,
                    forward=lambda y, x: (y + 1, x   ),
                    left   =lambda y, x: (y    , x + 1),
                    right  =lambda y, x: (y    , x - 1),
                ),
                Location(
                    angle=270,
                    forward=lambda y, x: (y    , x - 1),
                    left   =lambda y, x: (y + 1, x    ),
                    right  =lambda y, x: (y - 1, x    ),
                ),
            ])

        async def __call__(self):
            maze = self.maze.maze
            while True:
                if (shutdown := self.power_cycle.send((self.power, self.state, self.total_ticks))):
                    self.power = AgentPower.Off
                    self.state = None
                if self.power is AgentPower.On:
                    match self.state:
                        case AgentState.Moving:
                            old_loc = self.location
                            new_loc = next(self._hold).forward(*old_loc)
                            if not (0 <= new_loc[0] < maze.shape[0] and 0 <= new_loc[-1] < maze.shape[-1]):
                                pass
                            elif maze[old_loc] is Tile.Exit:
                                self.location = old_loc
                            elif maze[new_loc] is Tile.Floor or maze[new_loc] is Tile.Exit:
                                self.location = new_loc
                        case AgentState.TurningLeft:
                            self.num_turns += 1
                            next(self._turn_left)
                        case AgentState.TurningRight:
                            self.num_turns += 1
                            next(self._turn_right)
                tick = rnd.uniform(.5 * self.tick, 1.5 * self.tick) if errors else self.tick
                logger.debug(
                    'Robot: tick=%f, total_ticks=%d, shutdown=%r, location=%r, angle=%r, state=%r, power=%r, @exit?=%r',
                    tick, self.total_ticks, shutdown, self.location, next(self._hold).angle, self.state, self.power, maze[self.location] is Tile.Exit,
                )
                self.total_ticks += 1
                await aio_sleep(tick)

        @property
        def handler(self):
            async def handler(reader, writer):
                while True:
                    req = Message.deserialize(await reader.read(1024))
                    resp = Response.Error()
                    if errors:
                        condition = rnd.choices(['stall', 'drop', 'disconnect', 'error', 'noerror'], weights=[5, 5, 5, 5, 80], k=1)[0]
                        logger.debug('handler condition %s', condition)
                        match condition:
                            case 'stall':
                                delay = rnd.uniform(0, 2)
                                logger.debug('handler stall for %f on %r', delay, req)
                                await aio_sleep(delay)
                            case 'drop':
                                logger.debug('handler drop message %r', req)
                                continue
                            case 'disconnect':
                                logger.debug('handler disconnect %r', req)
                                return
                            case 'error':
                                logger.debug('handler error %r', req)
                                writer.write(resp.serialize())
                                continue

                    match req:
                        case Request.PowerOn():
                            self.power = AgentPower.On
                            resp = Response.PoweredOn()
                        case Request.PowerOff():
                            self.power = AgentPower.Off
                            resp = Response.PoweredOff()
                        case Request.Test():
                            resp = Response.TestSuccess()
                        case Request.Move() if self.power is AgentPower.On:
                            if self.state is None:
                                self.prev_location = self.location
                                self.state = AgentState.Moving
                                resp = Response.MovingStart()
                        case Request.TurnLeft() if self.power is AgentPower.On:
                            if self.state is None:
                                self.num_turns = 0
                                self.state = AgentState.TurningLeft
                                resp = Response.TurningStart()
                        case Request.TurnRight() if self.power is AgentPower.On:
                            if self.state is None:
                                self.num_turns = 0
                                self.state = AgentState.TurningRight
                                resp = Response.TurningStart()
                        case Request.StopMove() if self.power is AgentPower.On:
                            if self.state is AgentState.Moving:
                                self.state = None
                                resp = Response.MovingStop()
                        case Request.StopTurn() if self.power is AgentPower.On:
                            if self.state is AgentState.TurningLeft or self.state is AgentState.TurningRight:
                                self.state = None
                                resp = Response.TurningStop()
                        case Request.CheckMove() if self.power is AgentPower.On:
                            if self.state is AgentState.Moving:
                                resp = Response.MovingState(abs(array(self.prev_location) - array(self.location)).sum())
                        case Request.CheckTurn() if self.power is AgentPower.On:
                            if self.state is AgentState.TurningLeft or self.state is AgentState.TurningRight:
                                resp = Response.TurningState(turns=self.num_turns)
                        case Request.FrontSensor() if self.power is AgentPower.On:
                            sensor_loc = next(self._hold).forward(*self.location)
                            if 0 <= sensor_loc[0] < self.maze.maze.shape[0] and 0 <= sensor_loc[-1] < self.maze.maze.shape[-1]:
                                resp = Response.Wall() if self.maze.maze[sensor_loc] is Tile.Wall else Response.NoWall()
                        case Request.LeftSensor() if self.power is AgentPower.On:
                            sensor_loc = next(self._hold).left(*self.location)
                            if 0 <= sensor_loc[0] < self.maze.maze.shape[0] and 0 <= sensor_loc[-1] < self.maze.maze.shape[-1]:
                                resp = Response.Wall() if self.maze.maze[sensor_loc] is Tile.Wall else Response.NoWall()
                        case Request.RightSensor() if self.power is AgentPower.On:
                            sensor_loc = next(self._hold).right(*self.location)
                            if 0 <= sensor_loc[0] < self.maze.maze.shape[0] and 0 <= sensor_loc[-1] < self.maze.maze.shape[-1]:
                                resp = Response.Wall() if self.maze.maze[sensor_loc] is Tile.Wall else Response.NoWall()
                        case Request.ExitSensor() if self.power is AgentPower.On:
                            resp = Message.Exit() if self.maze.maze[self.location] is Tile.Exit else Message.NoExit()
                    writer.write(resp.serialize())
            return handler

    async def server(maze):
        ag = Agent(
            maze=maze,
            state=None,
            location=maze.start,
            tick=tick,
            power_cycle=
                pumped(lambda: (x for x in repeat(False)))()
                if power_cycle_freq is None else
                pumped(lambda: (x for x in cycle(chain(repeat(False, power_cycle_freq-1), [True]))))(),
        )
        server = await start_server(ag.handler, host, port)
        for name in {sock.getsockname() for sock in server.sockets}:
            logger.debug('Serving on: %r', name)
        async with TaskGroup() as tg:
            tg.create_task(ag())
            async with server:
                await server.serve_forever()

    run(server(maze=maze))

@contextmanager
def connection(host, port):
    sock = socket(AF_INET,SOCK_STREAM)
    sock.connect((host, port))
    def send(req : Request) -> Response:
        if not isinstance(req, Request):
            raise TypeError(f'req must be of type {Request} was {type(req)}')
        sock.sendall(req.serialize())
        resp = Message.deserialize(sock.recv(1024))
        if not isinstance(resp, Response):
            raise TypeError(f'resp must be of type {Response} was {type(resp)}')
        return resp
    yield send

parser = ArgumentParser()
parser.add_argument('--standalone', action='store_true', default=False, help='run mazerunner agent standalone')
parser.add_argument('--errors', action='store_true', default=False, help='enable errors')
parser.add_argument('--host', type=str, default='127.0.0.1', help='TCP host for mazerunner agent')
parser.add_argument('--port', type=int, default=8855, help='TCP port for mazerunner agent')
parser.add_argument('--maze', required=True, type=Path, help='path to maze')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--tick', type=float, default=1, help='tick speed')
parser.add_argument('--power-cycle-freq', type=int, default=None, help='power cycle frequency')
parser.add_argument('-v', '--verbose', action='count', default=0, help='increase logging verbosity')

if __name__ == '__main__':
    args = parser.parse_args()
    basicConfig(level={0: INFO, 1: DEBUG}.get(args.verbose, INFO))

    agent_kwargs= {
        'host': args.host, 'port': args.port,
        'maze': args.maze, 'seed': args.seed, 'errors': args.errors,
        'tick': args.tick, 'power_cycle_freq': args.power_cycle_freq,
    }
    if args.standalone:
        exit(agent_process(**agent_kwargs))
    else:
        (proc := Process(target=agent_process, kwargs=agent_kwargs)).start()

        @atexit_register
        def kill_agent_target(proc=proc):
            logger.debug('Killing %d', proc.pid)
            kill(proc.pid, SIGTERM)
            proc.join()
        sleep(.1)


    def directions():
        rotation = { "up":    (-1, 0),
                     "left":  ( 0,-1),
                     "down":  ( 1, 0),
                     "right": ( 0, 1),
        }
        previous = "up"
        reversing = "False"
        while True:
            while True:
                for direction, value in rotation.items():
                    if reversing is True:
                        if (reversing := previous != direction):
                            continue
                    previous = direction
                    reverse = yield direction, value
                    if reverse is True:
                        break
                if reverse is True:
                    reversing = True
                    break
            while True:
                for direction, value in reversed(rotation.items()):
                    if reversing is True:
                        if (reversing := previous != direction):
                            continue
                    previous = direction
                    reverse = yield direction, value
                    if reverse is True:
                        break
                if reverse is True:
                    reversing = True
                    break


    class RobotState():
        def __init__(self, x=0, y=0, reverse_turn=False):
            self.x = x
            self.y = y
            self.turning = "left"
            self.turns_history = [0]
            self.path_trace = [(x,y)]
            self.visited = set()
            self.reverse_turn = reverse_turn
            self._directions = directions()
            self._direction = next(self._directions)
            self.facing = self._direction[0]
            self.x_mov  = self._direction[1][0]
            self.y_mov  = self._direction[1][1]

        def change_direction(self, turns):
            if self.reverse_turn:
                self._directions.send(self.reverse_turn)
                if self.turning == "left":
                    self.turning = "right"
                else:
                    self.turning = "left"
                self.reverse_turn = False

            for _ in range(turns):
                self._direction = next(self._directions)
                self.facing = self._direction[0]
                self.x_mov  = self._direction[1][0]
                self.y_mov  = self._direction[1][1]
                self.turns_history[-1] += 1

        def position(self):
            return (self.x,self.y)

        def move_robot(self, steps):
            for _ in range(steps):
                self.x += self.x_mov
                self.y += self.y_mov

        def check_history(self, x, y):
            return (x, y) in self.visited

        def add_history(self, x, y, turns):
            self.visited.add((x,y))
            self.turns_history.append(turns)
            self.path_trace.append((x,y))

        def check_move(self):
            return (self.x + self.x_mov, self.y + self.y_mov)

        def backtrack(self):
            if not self.turns_history or not self.path_trace:
                print("Cannot backtrack further than the starting point!")
            elif self.turns_history and self.path_trace:
                self.turns_history.pop()
                self.path_trace.pop()


    class RobotCommands():
        def __init__(self, robot_sleep):
            self.robot_sleep = robot_sleep

        @staticmethod
        def power_on():
            return Request.PowerOn()

        @staticmethod
        def power_off():
            return Request.PowerOff()

        @staticmethod
        def check_exit(resp=None):
            return Request.ExitSensor()

        @staticmethod
        def exit_reached(resp=None):
            if isinstance(resp, Response.Exit):
                return True
            return False

        @staticmethod
        def check_wall():
            return Request.FrontSensor()

        @staticmethod
        def wall_hit(resp=None):
            if isinstance(resp, Response.Wall):
                return True
            return False

        @staticmethod
        def test_connection():
            return Request.Test()

        def move_unit(self, units=1):
            yield Request.Move()
            resp = yield Request.CheckMove()
            while resp.distance < units:
                yield Sleep(self.robot_sleep)
                resp = yield Request.CheckMove()
            yield Request.StopMove()

        def turn_left(self, turns=1):
            yield Request.TurnLeft()
            resp = yield Request.CheckTurn()
            while resp.turns < turns:
                yield Sleep(self.robot_sleep)
                resp = yield Request.CheckTurn()
            yield Request.StopTurn()

        def turn_right(self, turns=1):
            yield Request.TurnRight()
            resp = yield Request.CheckTurn()
            while resp.turns < turns:
                yield Sleep(self.robot_sleep)
                resp = yield Request.CheckTurn()
            yield Request.StopTurn()

        def turnaround(self):
            yield from self.turn_right(2)


    class Sleep:
        def __init__(self, delay):
            self.delay = delay


    def RobotSolver(Robot, Commands):
        while True:
            resp = yield Commands.check_exit()
            if Commands.exit_reached(resp):
                while True:
                    yield "Maze Completed"
            resp = yield Commands.check_wall()

            if not Commands.wall_hit(resp) and not Robot.check_history(*Robot.check_move()):
                logger.info("Moving...")
                Robot.move_robot(1)
                Robot.add_history(*Robot.position(), turns=0)
                yield from Commands.move_unit(1)
            else:
                logger.info("Encountered a wall or already-seen cell, try turning...")
                logger.info("Turning left...")
                Robot.change_direction(1)
                yield from Commands.turn_left()

                if Robot.turns_history[-1] >= 4:
                    logger.info(f"Tried every direction at {Robot.position()} - backtracking...")
                    # use right turns when backtracking to level the wear on gears somewhat
                    if Robot.turning == "left":
                        Robot.reverse_turn = True

                    logger.info("Turning back...")
                    Robot.change_direction(2)
                    yield from Commands.turnaround()

                    logger.info("Moving back...")
                    Robot.move_robot(1)
                    yield from Commands.move_unit(1)

                    logger.info("Resetting direction...")
                    Robot.change_direction(2)
                    yield from Commands.turnaround()

                    if Robot.turning == "right":
                        Robot.reverse_turn = True
                    Robot.backtrack()


    from functools import wraps
    def manage_power(func):
        @wraps(func)
        def inner(*args, **kwargs):
            try:
                resp = func(*args, **kwargs)
            except Exception as e_outer:
                status = func(args[0], Request.PowerOn())
                if not isinstance(status, Response.PoweredOn):
                    raise Exception("Failed to power on...") from e_outer
                resp = func(*args, **kwargs)
            if isinstance(resp, Response.Exit):
                logger.info("Powering off...")
                status = func(args[0], Request.PowerOff())
                if not isinstance(status, Response.PoweredOff):
                    raise Exception("Failed to power off...")
                logger.info("Powered off...")
            return resp
        return inner


    @manage_power
    def command_runner(client, req):
        if isinstance(resp := client(req), Response.Error):
            logger.debug(f"Received Error response for request: {req}")
            raise Exception
        logger.debug(f"Sending request: {req} -> Receiving response {resp}")
        return resp


    ### YOUR WORK HERE ###
    with connection(host=args.host, port=args.port) as send:

        Robot = RobotState(0,0)
        Robot.visited.add(Robot.position())
        Commands = RobotCommands(0.002)
        Solution = RobotSolver(Robot, Commands)
        next_step = next(Solution)
        resp = command_runner(send, Commands.test_connection())
        #resp = command_runner(send, Commands.power_on()) ### Remove

        while True:
            match next_step:
                case Request():
                    resp = command_runner(send, next_step)
                case Sleep():
                    sleep(next_step.delay)
                    logger.debug(f"Sleeping for {next_step.delay}")
                case "Maze Completed":
                    break
            next_step = Solution.send(resp)
            logger.debug(f"Current robot position is: {Robot.position()}")

        logger.info("Maze complete.")

