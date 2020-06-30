#!/usr/bin/env python3

from tkinter import *
from tkinter.font import Font


COLORS = ['black', 'white']


class Visualizer(object):

    def __init__(self, board_size, unit_length, text_size=15):
        self._board_size = board_size
        self._ulen = unit_length
        self._root = Tk()
        self._width = self._height = (1 + board_size) * self._ulen
        self._txt_font = Font(size=text_size)
        self.w = Canvas(
            self._root,
            width=self._width,
            height=self._height,
            bg='#EEE8AA'
        )
        self._draw_grid()
        self.w.pack()

    def _draw_grid(self):
        self.w.create_rectangle(
            self._ulen,
            self._ulen,
            self._width - self._ulen,
            self._height - self._ulen,
            fill="#EEE8AA"
        )
        for i in range(1, self._board_size):
            # draw horizontal lines
            self.w.create_line(
                self._ulen,
                self._ulen + i * self._ulen,
                self._width - self._ulen,
                self._ulen + i * self._ulen,
                fill="black"
            )
            # draw vertical lines
            self.w.create_line(
                self._ulen + i * self._ulen,
                self._ulen,
                self._ulen + i * self._ulen,
                self._height - self._ulen,
                fill="black"
            )

    def visualize(self, game_records):
        for idx, point in enumerate(game_records):
            self.w.create_oval(
                self._ulen + point[1]*self._ulen,
                self._ulen + point[0]*self._ulen,
                self._ulen + point[1]*self._ulen,
                self._ulen + point[0]*self._ulen,
                fill=COLORS[idx%2],
                outline=COLORS[idx%2],
                width=30
            )
            self.w.create_text(
                self._ulen + point[1]*self._ulen,
                self._ulen + point[0]*self._ulen,
                text=str((idx+2)//2),
                fill=COLORS[(idx+1)%2],
                font=self._txt_font
            )
        mainloop()


if __name__ == '__main__':
    import sys
    v = Visualizer(15, 50)
    game_recs = eval(open(sys.argv[1]).read())
    v.visualize(game_recs)
