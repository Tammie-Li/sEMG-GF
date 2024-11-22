from Plot.waveform import Visualization


class PlotManager:
    def __init__(self):
        self.visualization = Visualization()

    def plot_waveform(self, x1, x2, x3, mode='line'):
        if mode == "line_group":
            self.visualization.plot1D_line_group(x1)
        elif mode == "line":
            self.visualization.plot1D_line(x1)
        elif mode == "2D_line_group":
            self.visualization.plot2D_line(x1, x2, x3)


