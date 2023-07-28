from typing import Tuple, List
import numpy as np
import cvxpy as cp
from constants import *
from board import Board


class MinesweeperLPSolver:
    """Solving Minesweeper using LP

    Use Linear Programming to solve minesweeper interactively

    Attributes
    ---------
    board : Board
        The board the solver is going to try to solve.
    verbose : int
        Controls the verbosity.
    x : cp.Variable
        Variables that go into the cvxpy problem. A value of one means that block (x, y) is a mine.
    constraints : List[cvxpy.constraints.constraint.Constraint]
        Linear Program constraints to be used for iterative solving.
    objective : cvxpy.problems.objective.Objective
        Zero objective that needs to be provided to the Linear Problem.
    last_solution : List[List[float]]
        Solution values from previous iteration, can be used to speed up computation of new solution.
    known : numpy.ndarray
        Number of mines in each revealed square or -1.
    clicked : set
        Which blocks have been clicked.
    flagged : set
        Which blocks have been flagged.
    prob : cvxpy.Problem
        CVXPY problem that is iteratively used.
    """

    def __init__(self, board: Board, verbose: int = 0) -> None:
        self.board = board
        self.verbose = verbose
        self.x: cp.Variable = cp.Variable(
            [self.board.windowWidth, self.board.windowHeight]
        )
        self.constraints: List[cp.constraints.constraint.Constraint] = [
            self.x >= 0,
            1 >= self.x,
            cp.sum(self.x) == self.board.mines,
        ]
        self.objective: cp.problems.objective.Objective = cp.Maximize(
            0
        )  # Fake objective
        self.last_solution: np.ndarray = np.empty(
            [self.board.windowHeight, self.board.windowWidth]
        )
        self.known = -np.ones(((self.board.windowHeight, self.board.windowWidth)))
        self.clicked = set()
        self.flagged = set()
        self.prob: cp.Problem = cp.Problem(self.objective, self.constraints)

    def get_next(self) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Returns the coordinates of the next blocks to click or flag

        Construct the problem, solve it, use the solution to decide on what blocks to click.

        Parameters
        ----------

        Returns
        -------
        zero_pos : List[Tuple[int, int]
            Blocks with the smallest probability to contain a mine that haven't been clicked yet.
        one_pos : List[Tuple[int, int]
            Blocks with probability close to enough to one that have not been flagged.
        """

        # Create and solve the problem
        self.prob = cp.Problem(self.objective, self.constraints)
        self.prob.solve()

        # Retrieve the solution and find which blocks are close enough to one
        # and which close enough to zero; or the unclicked one with lowest value.
        zero_pos = []
        one_pos = []

        min_val = 2
        min_pos = None
        for i in range(self.board.windowWidth):
            for j in range(self.board.windowHeight):
                curr = self.x[i, j].value
                if curr is not None:
                    if curr < min_val and (i, j) not in self.clicked:
                        min_val = curr
                        min_pos = (i, j)
                    if curr <= 0.01 and (i, j) not in self.clicked:
                        zero_pos.append((i, j))
                    elif curr >= 0.99 and (i, j) not in self.flagged:
                        one_pos.append((i, j))
                        self.flagged.add((i, j))

        if not zero_pos:
            zero_pos.append(min_pos)
        return zero_pos, one_pos

    def add_constraint(self, i: int, j: int, mine_count: int) -> None:
        """Add a constraint based on the value of block (i, j)

        Parameters
        ----------
        i : int
            Which column the block is in. #NOTE: it seems to be flipped actually?
        j : int
            Which row the block is in.
        mine_count : int
            How many mines there are in the block (i, j).

        Returns
        -------
        """
        if self.verbose > 2:
            print("Adding", i, j, mine_count)

        # Add a constraint for this block and its neighbors
        self.clicked.add((i, j))

        coordinate_dict = {}
        if i >= 1:
            coordinate_dict["top_middle"] = (i - 1, j)
        if i + 1 < self.board.windowWidth:
            coordinate_dict["bottom_middle"] = (i + 1, j)
        if j >= 1:
            coordinate_dict["left"] = (i, j - 1)
            if i >= 1:
                coordinate_dict["top_left"] = (i - 1, j - 1)
            if i + 1 < self.board.windowWidth:
                coordinate_dict["bottom_left"] = (i + 1, j - 1)
        if j + 1 < self.board.windowHeight:
            coordinate_dict["right"] = (i, j + 1)
            if i >= 1:
                coordinate_dict["top_right"] = (i - 1, j + 1)
            if i + 1 < self.board.windowWidth:
                coordinate_dict["bottom_right"] = (i + 1, j + 1)

        for key, value in coordinate_dict.items():
            row, col = value
            coordinate_dict[key] = self.x[row, col]

        constraint = mine_count == cp.sum(list(coordinate_dict.values()))

        self.constraints.append(constraint)
        self.constraints.append(self.x[i, j] == 0)
