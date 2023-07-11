"""Minesweeper

This program uses Linear Programming to solve the classic game of minesweeper.
Through the use of pygame the game state can be visualized.
With minimal changes it can become a functional minesweeper game with mouse controls.

The three classes are `Board` which stores the state and is responsible for drawing anything game related.
The `MinesweeperLPSolver` class is responsible for computing a set of points to click and/or flag in each iteration.
Finally, `Game` handles the communication between the two and incorporates the main loop of the game, including user controls.

University of Chicago
CMSC 27200, Theory of Algorithms
Spring 2023
Konstantinos Ameranis
"""

import argparse
import time
from typing import Union
import pygame
from pygame.locals import *
from board import Board
from solver import MinesweeperLPSolver
from constants import *

class Game:
    """Game

    The game module is responsible for running the game.

    Attributes
    ----------
    windowWidth : int
        Number of columns in the board.
    windowHeight : int
        Number of rows in the board.
    blockSize : int
        Size of eachblock in pixels.
    board : Board
        Board object that stores all the minesweeper information.
    mines : int
        Number of mines on the board.
    seed : Union[None, int]
        Random seed to recreate the same boards.
    render : bool
        Whether to render the board on the screen. Set to False for testing and when running in an environment with no graphical environment.
    verbose : int
        Level of verbosity.
    time : float
        Timestamp the current game started.
    win_time : float
        Time it has taken to end the game, win or lose.
    solver : MinesweperLPSolver
        Solver used to uncover the solution to the game.
    """
    def __init__(self, windowWidth: int = COLUMNS, windowHeight: int = ROWS, blockSize: int = BLOCK_SIZE, mines: int = MINES, seed: Union[None, int] = None, render: bool = True, verbose: int = 0) -> None:
        """This is the initializing function of the Game object"""
        self.windowWidth = windowWidth
        self.windowHeight = windowHeight
        self.blockSize = blockSize
        self.board = Board(self.windowWidth, self.windowHeight, self.blockSize, mines, seed, render, verbose)
        self.render = render
        self.verbose = verbose
        self.time: float = 0
        self.win_time: float = 0
        self.font = None
        self._running = False
        self._display_surf: pygame.Surface
        self.solver: MinesweeperLPSolver
        self.loop_count = 0

    def on_init(self) -> None:
        """Start a new game

        Have the board create a new instance and create a new solver object.

        Parameters
        ----------

        Returns
        -------
        """
        self._running = False
        if self.render:
            pygame.init()
            self._display_surf = pygame.display.set_mode((self.windowWidth * self.blockSize, self.windowHeight * self.blockSize + 80), pygame.HWSURFACE)
            pygame.display.set_caption('UChicago TheoryWorld Minesweeper')
            self.font = pygame.font.SysFont("Times New Roman", 40)
        self.board.create()
        self.solver = MinesweeperLPSolver(self.board, verbose=self.verbose)
        self.time = time.time()
        self.win_time = 0
        self.loop_count = 0

        self._running = True

    def on_loop(self) -> None:
        """Actions taken every round

        Parameters
        ----------

        Returns
        -------
        """
        if self.board._bombed:
            self._running = False
            self.win_time = time.time()
            return
        click, flag = self.solver.get_next()
        for x, y in click:
            revealed_squares = self.board.click(x, y)
            for i, j, mine_count in revealed_squares:
                self.solver.add_constraint(i, j, mine_count)

        for x, y in flag:
            if self.verbose > 1:
                print("Flagging", x, y)
            self.board.flag(x, y)

        self.loop_count += 1
        if self.verbose > 0:
            progress_percent = 1 - self.board.remaining / (self.windowHeight * self.windowWidth - self.board.mines)
            progress_bar = '▉' * int(progress_percent * 40)
            print(f'Loop Count {self.loop_count:3d}: Blocks remaining = {self.board.remaining:4d} [{progress_bar:40s}]')

    def on_render(self) -> None:
        """Render the board in pygame

        Reset the surface, have the board render itself, print the extra information.

        Parameters
        ----------

        Returns
        -------
        """
        if not self.render:
            return
        self._display_surf.fill((0, 128, 255))
        self.board.draw(self._display_surf)
        if self._running:
            time_text = self.font.render(f'{time.time() - self.time:.0f}s', False, (255, 255, 255))
        else:
            time_text = self.font.render(f'{self.win_time - self.time:.0f}s', False, (255, 255, 255))
        self._display_surf.blit(time_text, (80, self.windowHeight * self.blockSize + 10))
        remaining_text = self.font.render(str(self.board.flags_remaining), False, (255, 255, 255))
        self._display_surf.blit(remaining_text, (self.windowWidth * self.blockSize - 80, self.windowHeight * self.blockSize + 10))
        pygame.display.flip()

    def win(self) -> None:
        """Win! Congratulations

        Stop the game and congratulate.

        Parameters
        ----------

        Returns
        -------
        """
        self.win_time = time.time()
        if self.render:
            self.on_render()
            text = pygame.font.SysFont('Times New Roman', 100)
            text_surf = text.render("Y O U   W I N !", True, (255, 0, 0), (0, 128, 255))
            textRect = text_surf.get_rect()
            textRect.center = (self.windowWidth * self.blockSize // 2, self.windowHeight * self.blockSize // 2)
            self._display_surf.blit(text_surf, textRect)
            pygame.display.flip()
        self._running = False

    def quit(self) -> None:
        """End the Game

        Performs cleanup.

        Parameters
        ----------

        Returns
        -------
        """
        if self.render:
            pygame.quit()

    def on_execute(self) -> bool:
        """Main loop of the game

        Receive input for control, get next solutions and render.

        Parameters
        ----------

        Returns
        -------
        won : bool
            Whether the last board was won.
        """
        self.on_init()

        while True:
            if self.render:
                pygame.event.pump()
                keys = pygame.key.get_pressed()

            if self._running:
                # If you fiddle with this comment you can turn this code into a 1-player minesweeper game.
                #events = pygame.event.get()
                #for event in events:
                #    if event.type == pygame.MOUSEBUTTONUP:
                #        x, y = pygame.mouse.get_pos()
                #        x, y = x // self.blockSize, y // self.blockSize
                #        if event.button == 1 and (x < self.windowWidth) and (y < self.windowHeight):       # Left click
                #            self.board.click(x, y)
                #        if event.button == 3:       # right click
                #            self.board.flag(x, y)
                if self.render:
                    self.on_render()
                self.on_loop()


                if self.board.has_won():
                    self.win()
            if self.render:
                if keys[K_SPACE]:
                    self.on_init()

                if keys[K_ESCAPE]:
                    self.quit()
                    return self.board.has_won()
            if not self.render and not self._running:
                return self.board.has_won()
            # time.sleep(1 / 50)
        self.quit()
        return self.board.has_won()


def parse_args() -> argparse.Namespace:
    """Parse arguments

    Use the `-h` option to list available command line arguments
    """
    parser = argparse.ArgumentParser(prog='Minesweeper Solver',
                                     description='You will be using Linear Programming iteratively to progressively solve a minesweeper board',
                                     epilog='University of Chicago, CMSC 27200 Spring \'23 Konstantinos Ameranis, Lorenzo Orecchia, Ivan Galakhov')
    parser.add_argument('--rows', '-r', type=int, default=ROWS, help='Board rows')
    parser.add_argument('--columns', '-c', type=int, default=COLUMNS, help='Board columns')
    parser.add_argument('--mines', '-m', type=int, default=MINES, help='Number of mines on the board')
    parser.add_argument('--seed', '-s', help='Provide a seed to recreate the same boards.')
    parser.add_argument('--no-render', action='store_true', help='Disable rendering for testing purposes')
    parser.add_argument('--block-size', '-b', type=int, default=BLOCK_SIZE, help='Size of each block in pixels')
    parser.add_argument('--verbose', '-v', action='count', default=0, help='Increase verbosity')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    game = Game(windowWidth=args.columns, windowHeight=args.rows, blockSize=args.block_size, mines=args.mines, seed=args.seed, render=not args.no_render, verbose=args.verbose)
    game.on_execute()
