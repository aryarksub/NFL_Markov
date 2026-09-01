class State:
    def __init__(self, down, ydstogo, yardline_100, covars = {}):
        self.down = down
        self.ydstogo = ydstogo
        self.yardline_100 = yardline_100
        self.covars = covars

    def __str__(self):
        return 'Down: {d}, Yard to Go: {y}, Yardline: {l}'.format(d = self.down, y = self.ydstogo, l = self.yardline_100)