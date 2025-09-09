import numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog, QSlider
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
import sys
import extract

def get_fps(fps):
    for factor in range(1, 80):
        if 40<= fps*factor <= 80:
            return fps*factor
        
    return fps




# Function to compute line intersections with plot boundaries
def compute_line_endpoints(x, y, angle, x_min=0, x_max=100, y_min=0, y_max=100):
    dx = np.cos(angle)
    dy = np.sin(angle)
    t_values = []
    if dx != 0:
        t_values.append((x_min - x) / dx)
        t_values.append((x_max - x) / dx)
    if dy != 0:
        t_values.append((y_min - y) / dy)
        t_values.append((y_max - y) / dy)
    t_values = [t for t in t_values if t > 0]
    t_min = min(t_values)
    x_end = x + t_min * dx
    y_end = y + t_min * dy
    return x_end, y_end




class RealTimePlotter(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget, world: extract.World):
        super().__init__()
        self.parent = parent
        self.world = world

        self.zoom = 1

        self.fps = world.fps
        self.dimx = world.dimx
        self.dimy = world.dimy
        self.plot_fps = get_fps(world.fps)
        print("Plot FPS:", self.plot_fps)
        self.update_interval = self.plot_fps / self.fps
        self.counter = 0

        
        self.entities = []


        # Initialize PyQtGraph plot
        self.plot_widget = pg.PlotWidget()
        # Add grid to the plot
        self.plot_widget.showGrid(x=True, y=True)
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.plot_widget)
        self.setLayout(self.layout)


            

        # Set plot background to white
        self.plot_widget.setBackground(None)

        # Configure plot
        self.plot_widget.setXRange(0, self.dimx, padding=0)
        self.plot_widget.setYRange(0, self.dimy, padding=0)
        # Disable zoom and pan
        self.plot_widget.setMouseEnabled(x=False, y=False)

        # Store plot items
        self.points_e_plot = pg.ScatterPlotItem(pxMode=False)



        # Add plot items to the plot widget
        self.plot_widget.addItem(self.points_e_plot)


        # Initialize timer for updates
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(int(1000/self.plot_fps / 2))  # 20 FPS





    


    def update_plot(self):
        if self.counter % self.update_interval == 0:
            self.state = self.world.get_state()
            self.entities = self.state.entities




        self.points_e_plot.setData(
            x = [e.x for e in self.entities],
            y = [e.y for e in self.entities],
            symbol='o',
            brush=[pg.mkBrush(0,0,0) for _ in self.entities],
            size=[e.size * 2 * self.zoom  for e in self.entities]
        )

        self.counter += 1



        
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, world: extract.World):
        super().__init__()

        self.setWindowTitle("evolvE")

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)

        self.pauseButton = QtWidgets.QPushButton("Pause")
        self.pauseButton.setCheckable(True)
        self.pauseButton.clicked.connect(self.toggle_pause)
        layout.addWidget(self.pauseButton)

        self.real_time_plotter = RealTimePlotter(self,  world)

        # Get the width and height of the widget in pixels
        width = self.real_time_plotter.size().width()
        height = self.real_time_plotter.size().height()

        if world.dimx > world.dimy:
            self.real_time_plotter.setFixedWidth(width)
            self.real_time_plotter.setFixedHeight(int(width * world.dimy / world.dimx))
        
        if world.dimx < world.dimy:
            self.real_time_plotter.setFixedWidth(int(height * world.dimx / world.dimy))
            self.real_time_plotter.setFixedHeight(height)
        
        layout.addWidget(self.real_time_plotter)




        
    def toggle_pause(self):
        """Toggle the paused state of the real-time plotter."""
        if self.pauseButton.isChecked():
            self.pauseButton.setText("Resume")
            self.real_time_plotter.timer.stop()
        else:
            self.pauseButton.setText("Pause")
            self.real_time_plotter.timer.start(int(1000 / self.real_time_plotter.plot_fps))

        


if __name__ == "__main__":
    app = QApplication([])
    file_dialog = QFileDialog()
    file_path, _ = file_dialog.getOpenFileName(None, "Select Binary File", "", "Binary Files (*.bin)")

    del app, file_dialog

    if not file_path:
        print("No file selected")
        exit()

    world = extract.World(file_path)

    print(world.fps, world.dimx, world.dimy, world.entity_bytes_0, world.store_capacity)
    

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(world)
    window.resize(800, 800)
    window.show()
    sys.exit(app.exec())
