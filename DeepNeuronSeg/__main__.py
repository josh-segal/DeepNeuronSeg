print("Launching DeepNeuronSeg...")

from PyQt5.QtWidgets import QApplication
from DeepNeuronSeg.controllers.app_controller import MainWindow

import sys

my_stylesheet = """
    QWidget {
        background-color: #2b2b2b;
        color: #ffffff;
        margin: 0px;
        padding: 0px;
    }
    
    QPushButton {
        background-color: #3d3d3d;
        border: 1px solid #555555;
        padding: 4px;
        margin: 1px;
        border-radius: 3px;
    }
    
    QPushButton:hover {
        background-color: #4f4f4f;
    }
    
    QPushButton:pressed {
        background-color: #2e2e2e;
    }
    
    QLineEdit, QTextEdit, QPlainTextEdit {
        background-color: #1e1e1e;
        border: 1px solid #3d3d3d;
        color: #ffffff;
        padding: 2px;
    }
    
    QComboBox {
        background-color: #3d3d3d;
        border: 1px solid #555555;
        border-radius: 3px;
        padding: 1px 18px 1px 3px;
        color: #ffffff;
    }
    
    QComboBox::drop-down {
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 15px;
        border-left-width: 1px;
        border-left-color: #555555;
        border-left-style: solid;
    }
    
    QMenuBar {
        background-color: #2b2b2b;
    }
    
    QMenuBar::item {
        background: transparent;
    }
    
    QMenuBar::item:selected {
        background: #3d3d3d;
    }
    
    QMenu {
        background-color: #2b2b2b;
        border: 1px solid #3d3d3d;
    }
    
    QMenu::item:selected {
        background-color: #3d3d3d;
    }

    QTabWidget::pane {
        background-color: #2b2b2b;
        border: 1px solid #3d3d3d;
        margin: 2px;
        padding: 2px;
    }

    QTabBar::tab {
        background-color: #2b2b2b;
        color: #ffffff;
        border: 1px solid #3d3d3d;
        padding: 4px;
        border-top-left-radius: 3px;
        border-top-right-radius: 3px;
    }

    QTabBar::tab:selected {
        background-color: #3d3d3d;
        border-bottom: 1px solid #2b2b2b;
    }

    QTabBar::tab:hover {
        background-color: #4f4f4f;
    }
    
    QVBoxLayout, QHBoxLayout {
        margin: 2px;
        padding: 2px;
        spacing: 2px;
    }
    
    QListWidget {
        margin: 2px;
    }
"""  

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(my_stylesheet)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())



