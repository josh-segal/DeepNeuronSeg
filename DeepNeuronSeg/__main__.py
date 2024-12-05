print("Launching DeepNeuronSeg...")

from PyQt5.QtWidgets import QApplication
from DeepNeuronSeg.controllers.app_controller import MainWindow

import sys

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
