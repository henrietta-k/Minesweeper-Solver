from typing import Tuple, List, Union
import pygame
from constants import *
import random

class Board:
    """Board

    This object holds the state of minesweeper, including which blockss have been clicked, flagged and/or contain a mine.
    Also responsible for drawing itself on the pygame.Surface provided by the game.

    Attributes
    ----------
    windowWidth : int
        Number of columns in the board.
    windowHeight : int
        Number of rows in the board.
    blockSize : int
        Side length of each block in pixels.
    pressed : List[List[bool]]
        Whether sqaure (i, j) has been pressed.
    flagged ; List[List[bool]]
        whether block (i, j) is flagged.
    mines : int
        Number of mines present in the board.
    seed : Union[None, int]
        Seed to be used for creating the same boards.
    render : bool
        Whehter the board is rendered. Mostly used for testing and when running on a system with no graphical environment.
    verbose : int
        How much the object is printing about the operations it's taking.
    neighboring_mine_count ; List[List[int]]
        How many mines there are adjacent to block (i, j).
    colors : List[Tuple[str, Tuple[int, int, int]]]
        Assigned colors to different mine count. Similar to original game.
    flag_image : pygame.Surface
        Image used for rendering a flagged block.
    bomb_image : pygame.Surface
        Image used for rendering a revealed bomb after loss.
    _bombed : bool
        Whether a bomb was clicked on the current board.
    remaining : int
        How many non bomb blockss are left unclicked.
    self.flags_remaining : int
        How many mines there are, minus the number of flagged blocks on the board.
    """
    def __init__(self, windowWidth: int = COLUMNS, windowHeight: int = ROWS, blockSize: int = BLOCK_SIZE, mines: int = MINES, seed: Union[None, int] = None, render: bool = True, verbose: int = 0) -> None:
        """Initialize the Board with rows, columns, mines, block size, whether the board is rendered and verbosity"""
        self.windowWidth = windowWidth
        self.windowHeight = windowHeight
        self.blockSize = blockSize
        self.pressed: List[List[bool]]
        self.flagged: List[List[bool]]
        self.mines = mines
        self.seed = seed
        self.render = render
        self.verbose = verbose
        self.is_mine: List[List[bool]]
        self.neighboring_mine_count: List[List[int]]
        self.colors = [
            ("White", (192, 192, 192)),
            ("Blue", (0, 0, 255)),
            ("Green", (0, 255, 0)),
            ("Red", (255, 0, 0)),
            ("Purple", (255, 0, 255)),
            ("Black", (64, 64, 64)),
            ("Maroon", (128, 0, 0)),
            ("Gray", (128, 128, 128)),
            ("Turquoise", (64, 224, 208))
        ]
        self.font = None
        self.flag_image = None
        self.mine_image = None
        self._bombed = False
        self.remaining = self.windowWidth * self.windowHeight - self.mines
        self.flags_remaining = self.mines
        random.seed(self.seed)

    def create(self) -> None:
        """Create a new board

        Selects `mines` blocks at random and computes the necessary auxiliary variables.

        Parameters
        ----------

        Returns
        -------
        """
        # Reset variables
        self.pressed = [[False] * self.windowWidth for _ in range(self.windowHeight)]
        self.flagged = [[False] * self.windowWidth for _ in range(self.windowHeight)]
        self.is_mine = [[False] * self.windowWidth for _ in range(self.windowHeight)]
        self.neighboring_mine_count = [[0] * self.windowWidth for _ in range(self.windowHeight)]
        self.remaining = self.windowWidth * self.windowHeight - self.mines
        self.flags_remaining = self.mines
        self._bombed = False

        if self.render:
            self.font = pygame.font.SysFont("Times New Roman", 30)

        # Select mined blocks
        candidates = [(x, y) for x in range(self.windowWidth) for y in range(self.windowHeight)]
        random.shuffle(candidates)
        mines_assigned = 0
        for x, y in candidates:
            if (x < 2 and y < 2) or (x < 2 and y > self.windowHeight - 3) or (x > self.windowWidth - 3 and y < 2) or (x > self.windowWidth - 3 and y > self.windowHeight - 3):
                continue
            self.is_mine[y][x] = True
            mines_assigned += 1
            if mines_assigned == self.mines:
                break

        # Compute auxiliary variables
        for x in range(self.windowWidth):
            for y in range(self.windowHeight):
                if self.is_mine[y][x]:
                    for new_x, new_y in [(x+1, y), (x+1, y+1), (x, y+1), (x-1, y+1), (x-1, y), (x-1, y-1), (x, y-1), (x+1, y-1)]:
                        if (0 <= new_x < self.windowWidth) and (0 <= new_y < self.windowHeight):
                            self.neighboring_mine_count[new_y][new_x] = self.neighboring_mine_count[new_y][new_x] + 1
        if self.verbose > 0:
            print(f'Successfully created new board {self.windowHeight}x{self.windowWidth} with {self.mines} mines.')

    def draw(self, surface: pygame.Surface) -> None:
        """Draw board

        For each block, draw what is needed, number, flag or bomb.
        Finally draw the grid

        Parameters
        ----------
        surface : pygame.Surface
            The surface to draw on.

        Returns
        -------
        """
        if not self.render:
            return
        for x in range(self.windowWidth):
            for y in range(self.windowHeight):
                if self.pressed[y][x]:
                    pygame.draw.rect(surface, self.colors[0][1], (x * self.blockSize, y * self.blockSize, self.blockSize, self.blockSize))
                    if self.neighboring_mine_count[y][x] > 0:
                        text = self.font.render(str(self.neighboring_mine_count[y][x]), True, self.colors[self.neighboring_mine_count[y][x]][1])
                        textRect = text.get_rect()
                        textRect.center = ((x + 0.5) * self.blockSize, (y + 0.5) * self.blockSize)
                        surface.blit(text, textRect)
                if self.flagged[y][x] and not self._bombed:
                    if self.flag_image is None:
                        self.flag_image = pygame.image.load("flag.png").convert()
                        self.flag_image = pygame.transform.scale(self.flag_image, (self.blockSize, self.blockSize))
                    surface.blit(self.flag_image, (x * self.blockSize, y * self.blockSize))
                if self.is_mine[y][x] and self._bombed:
                    if self.mine_image is None:
                        self.mine_image = pygame.image.load("mine.jpg").convert()
                        self.mine_image = pygame.transform.scale(self.mine_image, (self.blockSize, self.blockSize))
                    surface.blit(self.mine_image, (x * self.blockSize, y * self.blockSize))
        for x in range(self.windowWidth):
            pygame.draw.rect(surface, (255, 255, 255), ((x + 1) * self.blockSize, 0, 2, self.windowHeight * self.blockSize))
        for y in range(self.windowHeight):
            pygame.draw.rect(surface, (255, 255, 255), (0, (y + 1) * self.blockSize, self.windowWidth * self.blockSize, 2))

    def click(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Click on block (x, y)

        Take care of revealing all necessary blocks and return them in a list

        Parameters
        ----------
        x : int
            Which column the block is in.
        y : int
            Which row the block is in.

        Returns
        -------
        revealed : List[Tuple[int, int]]
            All the points that are revealed by clicking on (x, y)
        """
        if self.verbose > 2:
            print("Clicked", x, y)
        if self._bombed or self.pressed[y][x] or self.flagged[y][x]:
            return []
        if self.is_mine[y][x]:
            self._bombed = True
            if self.verbose > 0:
                print("You goofed")
            return [(x, y, -1)]
        self.pressed[y][x] = True
        self.remaining = self.remaining - 1
        revealed = [(x, y, self.neighboring_mine_count[y][x])]

        # If there are no neighboring mines recurse on all neighbors
        if self.neighboring_mine_count[y][x] == 0:
            for new_x, new_y in [(x+1, y), (x+1, y+1), (x, y+1), (x-1, y+1), (x-1, y), (x-1, y-1), (x, y-1), (x+1, y-1)]:
                if (0 <= new_x < self.windowWidth) and (0 <= new_y < self.windowHeight):
                    revealed.extend(self.click(new_x, new_y))
        return revealed

    def flag(self, x: int, y: int) -> None:
        """Flag block (x, y)

        Informs the board that block (x, y) is flagged.

        Parameters
        ----------
        x : int
            Which column the block is in.
        y : int
            Which row the block is in.

        Returns
        -------
        """
        if self._bombed or self.pressed[y][x]:
            return
        self.flagged[y][x] = not self.flagged[y][x]
        self.flags_remaining += 1 if not self.flagged[y][x] else -1


    def has_won(self) -> int:
        """Whether the board is won

        Parameters
        ----------

        Returns
        -------
        won : bool
            Whether all non mined blocks have been clicked.
        """
        return self.remaining == 0
