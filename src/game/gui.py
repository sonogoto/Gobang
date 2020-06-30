#!/usr/bin/env python3

from tkinter import *
from tkinter import messagebox
from .exceptions import GameTied
from .player import BasePlayer

COLOR = {
    BasePlayer.WHITE: "white",
    BasePlayer.BLACK: "black"
}


class Gui(object):

    UNIT_LENGTH = 50

    def __init__(self, game):
        self._game = game
        self._width = (1 + game.board.board_size) * Gui.UNIT_LENGTH
        self._height = (1 + game.board.board_size) * Gui.UNIT_LENGTH
        self.w = Canvas(
            Tk(),
            width=self._width,
            height=self._height,
            bg='#EEE8AA'
        )
        self.w.pack()
        self._draw_grid()
        self.w.bind("<Button-1>", self._play)
        self.w.mainloop()

    def _draw_grid(self):
        self.w.create_rectangle(
            Gui.UNIT_LENGTH,
            Gui.UNIT_LENGTH,
            self._width - Gui.UNIT_LENGTH,
            self._height - Gui.UNIT_LENGTH,
            fill="#EEE8AA"
        )
        for i in range(1, self._game.board.board_size):
            # draw horizontal lines
            self.w.create_line(
                Gui.UNIT_LENGTH,
                Gui.UNIT_LENGTH + i * Gui.UNIT_LENGTH,
                self._width - Gui.UNIT_LENGTH,
                Gui.UNIT_LENGTH + i * Gui.UNIT_LENGTH,
                fill="black"
            )
            # draw vertical lines
            self.w.create_line(
                Gui.UNIT_LENGTH + i * Gui.UNIT_LENGTH,
                Gui.UNIT_LENGTH,
                Gui.UNIT_LENGTH + i * Gui.UNIT_LENGTH,
                self._height - Gui.UNIT_LENGTH,
                fill="black"
            )
        
    def _play(self, event):
        try:
            row_idx, col_idx = self._game.play(
                event=event,
                unit_length=Gui.UNIT_LENGTH
            )
            self._place(row_idx, col_idx)
        except GameTied:
            self.w.unbind("<Button-1>")
            messagebox.showinfo("Game Tied", "Game Tied")
        except RuntimeError:
            self.w.unbind("<Button-1>")
            messagebox.showinfo(
                "Game Over",
                "Game Over!\n Winner is %s" % COLOR[self._game.get_winner().color].upper()
            )

    def _place(self, row_idx, col_idx):
        self.w.create_oval(
            Gui.UNIT_LENGTH + col_idx*Gui.UNIT_LENGTH,
            Gui.UNIT_LENGTH + row_idx*Gui.UNIT_LENGTH,
            Gui.UNIT_LENGTH + col_idx*Gui.UNIT_LENGTH,
            Gui.UNIT_LENGTH + row_idx*Gui.UNIT_LENGTH,
            fill=COLOR[self._game.game_state[row_idx, col_idx]],
            outline=COLOR[self._game.game_state[row_idx, col_idx]],
            width=30
        )
