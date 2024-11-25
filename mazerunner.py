#!/usr/bin/env python3

from argparse import ArgumentParser
from asyncio import run, start_server, TaskGroup, sleep as aio_sleep
from atexit import register as atexit_register
from collections import namedtuple
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
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
    Move StopMove CheckMove
    TurnLeft TurnRight StopTurn CheckTurn
    FrontSensor LeftSensor RightSensor ExitSensor Test
''').strip().split():
    globals()[msg] = dataclass(frozen=True)(type(msg, (Request,), {}))
for msg in dedent('''
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

def agent_process(host, port, maze, seed, errors, tick):
    rnd = Random(seed)

    maze = Maze.from_file(maze)
    for row in overlay(render_maze(maze), {maze.start: '@'}):
        logger.debug('Maze: %s', ' '.join(row))

    AgentState = Enum('AgentState', 'Moving TurningLeft TurningRight')

    Location = namedtuple('Location', 'angle forward left right')

    @dataclass
    class Agent:
        state : AgentState
        location : tuple[int, int]
        maze : Maze

        tick : int = 1

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
                    'Robot: tick=%f, location=%r, angle=%r, state=%r, @exit?=%r',
                    tick, self.location, next(self._hold).angle, self.state, maze[self.location] is Tile.Exit,
                )
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
                        case Request.Test():
                            resp = Response.TestSuccess()
                        case Request.Move():
                            if self.state is None:
                                self.prev_location = self.location
                                self.state = AgentState.Moving
                                resp = Response.MovingStart()
                        case Request.TurnLeft():
                            if self.state is None:
                                self.num_turns = 0
                                self.state = AgentState.TurningLeft
                                resp = Response.TurningStart()
                        case Request.TurnRight():
                            if self.state is None:
                                self.num_turns = 0
                                self.state = AgentState.TurningRight
                                resp = Response.TurningStart()
                        case Request.StopMove():
                            if self.state is AgentState.Moving:
                                self.state = None
                                resp = Response.MovingStop()
                        case Request.StopTurn():
                            if self.state is AgentState.TurningLeft or self.state is AgentState.TurningRight:
                                self.state = None
                                resp = Response.TurningStop()
                        case Request.CheckMove():
                            if self.state is AgentState.Moving:
                                resp = Response.MovingState(abs(array(self.prev_location) - array(self.location)).sum())
                        case Request.CheckTurn():
                            if self.state is AgentState.TurningLeft or self.state is AgentState.TurningRight:
                                resp = Response.TurningState(turns=self.num_turns)
                        case Request.FrontSensor():
                            sensor_loc = next(self._hold).forward(*self.location)
                            if 0 <= sensor_loc[0] < self.maze.maze.shape[0] and 0 <= sensor_loc[-1] < self.maze.maze.shape[-1]:
                                resp = Response.Wall() if self.maze.maze[sensor_loc] is Tile.Wall else Response.NoWall()
                        case Request.LeftSensor():
                            sensor_loc = next(self._hold).left(*self.location)
                            if 0 <= sensor_loc[0] < self.maze.maze.shape[0] and 0 <= sensor_loc[-1] < self.maze.maze.shape[-1]:
                                resp = Response.Wall() if self.maze.maze[sensor_loc] is Tile.Wall else Response.NoWall()
                        case Request.RightSensor():
                            sensor_loc = next(self._hold).right(*self.location)
                            if 0 <= sensor_loc[0] < self.maze.maze.shape[0] and 0 <= sensor_loc[-1] < self.maze.maze.shape[-1]:
                                resp = Response.Wall() if self.maze.maze[sensor_loc] is Tile.Wall else Response.NoWall()
                        case Request.ExitSensor():
                            resp = Message.Exit() if self.maze.maze[self.location] is Tile.Exit else Message.NoExit()
                    writer.write(resp.serialize())
            return handler

    async def server(maze):
        ag = Agent(maze=maze, state=None, location=maze.start, tick=tick)
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
parser.add_argument('-v', '--verbose', action='count', default=0, help='increase logging verbosity')

if __name__ == '__main__':
    args = parser.parse_args()
    basicConfig(level={0: INFO, 1: DEBUG}.get(args.verbose, INFO))

    agent_kwargs= {
        'host': args.host, 'port': args.port,
        'maze': args.maze, 'seed': args.seed, 'errors': args.errors,
        'tick': args.tick,
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

    ### YOUR WORK HERE ###
    with connection(host=args.host, port=args.port) as send:
        while (resp := send(req := Request.ExitSensor())) != Response.Exit():
            logger.info('Request → Response: %16r → %r', req, resp)
            
            resp = send(req := Request.TurnRight())
            logger.info('Request → Response: %16r → %r', req, resp)

            while (resp := send(req := Request.FrontSensor())) != Response.NoWall():
                logger.info('Request → Response: %16r → %r', req, resp)

                resp = send(req := Request.CheckTurn())
                logger.info('Request → Response: %16r → %r', req, resp)

                sleep(1)

            resp = send(req := Request.StopTurn())
            logger.info('Request → Response: %16r → %r', req, resp)    
            
            resp = send(req := Request.Move())
            logger.info('Request → Response: %16r → %r', req, resp)

            sleep(2)

            resp = send(req := Request.CheckMove())
            logger.info('Request → Response: %16r → %r', req, resp)

            resp = send(req := Request.StopMove())
            logger.info('Request → Response: %16r → %r', req, resp)

        resp = send(req := Request.ExitSensor())
        logger.info('Request → Response: %16r → %r', req, resp)
