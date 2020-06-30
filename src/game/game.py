#!/usr/bin/env python3

from .player import BasePlayer
from .exceptions import NoSpaceLeft
from .exceptions import GameTied


class Game(object):

    MIN_CONTINUOUS_LENGTH = 5
    
    def __init__(self, board, player1, player2):
        self._board = board
        self._players = [player1, player2]
        self._current_player = 0 if player1.color == BasePlayer.BLACK else 1
        self._gameover = False

    @property
    def board(self):
        return self._board

    @property
    def players(self):
        return self._players

    @property
    def current_player(self):
        return self._players[self._current_player]

    @property
    def last_player(self):
        return self._players[1-self._current_player]

    @property
    def gameover(self):
        return self._gameover

    @property
    def gametied(self):
        return 0 not in self.game_state

    @property
    def game_state(self):
        return self._board.board_state

    def play(self, **kwargs):
        if self._gameover:
            raise RuntimeError('Game Is Over')
        try:
            row_idx, col_idx = self._players[self._current_player].play(
                self._board,
                **kwargs
            )
        except NoSpaceLeft:
            raise GameTied('Game Tied')
        self._board.place(
            row_idx,
            col_idx,
            self._players[self._current_player].color
        )
        self._update_gameover(row_idx, col_idx)
        if not self._gameover:
            self._swap_player()
        return row_idx, col_idx

    def _swap_player(self):
        self._current_player = 1 - self._current_player

    def _update_gameover(self, row_idx, col_idx):

        def _check_line(line, color):
            for k in range(line.__len__() - (Game.MIN_CONTINUOUS_LENGTH-1)):
                if sum(line[k:k+Game.MIN_CONTINUOUS_LENGTH]) \
                        == color * Game.MIN_CONTINUOUS_LENGTH:
                    return True
            return False

        def _calc_term_points():
            return (
                max(row_idx - (Game.MIN_CONTINUOUS_LENGTH-1), 0),
                max(col_idx - (Game.MIN_CONTINUOUS_LENGTH-1), 0),
                min(row_idx + Game.MIN_CONTINUOUS_LENGTH, self._board.board_size),
                min(col_idx + Game.MIN_CONTINUOUS_LENGTH, self._board.board_size)
            )

        def _get_check_region():
            return (
                self.game_state[row_idx, column_st:column_end],
                self.game_state[row_st:row_end, col_idx],
                [self.game_state[row_idx + k, col_idx + k]
                 for k in range(
                    max(row_st - row_idx, column_st - col_idx),
                    min(row_end - row_idx, column_end - col_idx))],
                [self.game_state[row_idx - k, col_idx + k]
                 for k in range(
                    max(row_idx - row_end + 1, column_st - col_idx),
                    min(row_idx - row_st + 1, column_end - col_idx))]
            )

        row_st, column_st, row_end, column_end = _calc_term_points()
        row, column, nw2se_diagonal, sw2ne_diagonal = _get_check_region()
        gameover = False
        if not gameover and _check_line(
                row,
                self.game_state[row_idx, col_idx]
        ):
            gameover = True
        if not gameover and _check_line(
                column,
                self.game_state[row_idx, col_idx]
        ):
            gameover = True
        if not gameover and _check_line(
                nw2se_diagonal,
                self.game_state[row_idx, col_idx]
        ):
            gameover = True
        if not gameover and _check_line(
                sw2ne_diagonal,
                self.game_state[row_idx, col_idx]
        ):
            gameover = True
        self._gameover = gameover

    def get_winner(self):
        assert self._gameover, \
            'GAME NOT OVER YET'
        return self.current_player

    def get_game_status(self):
        return {
            'game_state': self.game_state.copy(),
            'gameover': self._gameover,
            'current_player': self._current_player
        }

    def reset(self, game_status):
        self.board.reset(game_status['game_state'])
        self._gameover = game_status['gameover']
        self._current_player = game_status['current_player']
