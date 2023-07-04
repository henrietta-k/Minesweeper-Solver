# Minesweeper-Solver

Minesweeper is a game in which clicking a block reveals how many mines there are surrounding it. A player loses once they have clicked a block containing a mine. The solver computes the probability that each block contains a mine using Linear Programming, thereby allowing the game to be "solved" and for a player to win. 

## Installation

To install the required packages:
```bash
python -m pip install -r requirements.txt
```
## Testing

Games can be tested using this command where the number of rows (-r), columns (-c), and mines (-m) can be customized.
```bash
python game.py -b 40 -v -r 24 -c 30 -m 150
```
