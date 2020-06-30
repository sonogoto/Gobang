#!/usr/bin/env python3

from .game import Game
from .player import RandomPlayer, ManualPlayer
from .board import Board
from .gui import Gui


if __name__ == '__main__':
    player1 = ManualPlayer(1, RandomPlayer.BLACK)
    player2 = RandomPlayer(2, RandomPlayer.WHITE)
    board = Board(board_size=15)
    game = Game(board, player1, player2)
    Gui(game)
