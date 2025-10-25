import numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog, QSlider
import pyqtgraph as pg # type: ignore
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui  # type: ignore
import sys
import extract

LIGANDS = True
LSIZE = 5

def get_fps(fps):
    for factor in range(1, 80):
        if 40<= fps*factor <= 80:
            return fps*factor
        
    return fps


## TODO make entities clickable to show info

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


        self.clicked_on = None

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
        self.points_l_plot = pg.ScatterPlotItem(pxMode=False)


        # highlight item for clicked entity (transparent fill, colored outline)
        self.highlight_plot = pg.ScatterPlotItem(pxMode=False)
        self.highlight_plot.setBrush(pg.mkBrush(0,0,0,0))
        self.highlight_plot.setPen(pg.mkPen(255,0,0, width=2))
        self.plot_widget.addItem(self.highlight_plot)

        # info label shown when an entity is selected
        self.info_label = QtWidgets.QLabel(self.plot_widget)
        self.info_label.setStyleSheet("background-color: rgba(255,255,255,230); padding:4px; border: 1px solid #333;")
        self.info_label.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.info_label.hide()

        # Add plot items to the plot widget
        self.plot_widget.addItem(self.points_e_plot)
        self.plot_widget.addItem(self.points_l_plot)


        # Initialize timer for updates
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(int(1000/self.plot_fps / 2))  # 20 FPS
    
        # connect mouse click on the plot to handler
        self.plot_widget.scene().sigMouseClicked.connect(self.on_click)

    # GPT 5 generated
    def on_click(self, event):
        """Handle mouse click on plot, find nearest entity and print its id."""
        # map scene position to plot coordinates
        pos = event.scenePos()
        vb = self.plot_widget.plotItem.vb
        view_pos = vb.mapSceneToView(pos)
        x_click = view_pos.x()
        y_click = view_pos.y()

        if not hasattr(self, "entities") or len(self.entities) == 0:
            # clear selection
            self.clicked_on = None
            self.highlight_plot.clear()
            self.info_label.hide()
            return

        closest = None
        closest_dist2 = float("inf")
        for e in self.entities:
            dx = e.x - x_click
            dy = e.y - y_click
            dist2 = dx*dx + dy*dy
            size = getattr(e, "size", 1.0)
            threshold2 = max((size * 2.0) ** 2, 9.0)  # squared threshold (min radius 3)
            if dist2 <= threshold2 and dist2 < closest_dist2:
                closest_dist2 = dist2
                closest = e

        if closest is not None:
            eid = getattr(closest, "id", None)
            print(f"Clicked entity id: {eid}")
            self.clicked_on = eid
        else:
            # clicked empty space => clear selection
            self.clicked_on = None
            self.highlight_plot.clear()
            self.info_label.hide()

    

    def update_plot(self):
        if self.counter % self.update_interval == 0:
            self.state = self.world.get_state()
            self.entities = self.state.entities
            self.ligands = self.state.ligands




        self.points_e_plot.setData(
            x = [e.x for e in self.entities],
            y = [e.y for e in self.entities],
            symbol='o',
            brush=[pg.mkBrush(0,0,0) for _ in self.entities],
            size=[e.size * 2  for e in self.entities]
        )

        if LIGANDS:
            self.points_l_plot.setData(
                x = [l.x for l in self.ligands],
                y = [l.y for l in self.ligands],
                symbol='o',
                brush=[pg.mkBrush(255,0,0) for _ in self.ligands],
                size=[0.1 * LSIZE for _ in self.ligands]
            )

        self.counter += 1

        # update highlight + info box if an entity is selected
        if getattr(self, "clicked_on", None) is not None:
            ent = next((e for e in self.entities if e.id == self.clicked_on), None)
            if ent is not None:
                self.highlight_plot.setData(
                    x=[ent.x],
                    y=[ent.y],
                    size=[max(ent.size * 2 + 4, 8)],
                    brush=[pg.mkBrush(0,0,0,0)],
                    pen=[pg.mkPen(255,0,0, width=2)],
                )
                # update and show info label
                self.info_label.setText(f"id: {ent.id}\n x: {ent.x:.2f}\n y: {ent.y:.2f}\n size: {ent.size:.2f}")
                self.info_label.adjustSize()
                # place label in top-left of plot area (10px padding)
                self.info_label.move(10, 10)
                self.info_label.show()
            else:
                # selected id not present anymore
                self.clicked_on = None
                self.highlight_plot.clear()
                self.info_label.hide()
        else:
            self.highlight_plot.clear()
            self.info_label.hide()


        
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
    ratio = world.dimy / world.dimx
    window.resize(800, int(800 * ratio))
    window.show()
    sys.exit(app.exec())
