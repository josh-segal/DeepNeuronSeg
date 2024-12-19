from PyQt5.QtCore import pyqtSignal, QPointF, Qt
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtWidgets import QLabel

class ImageLabel(QLabel):
    """Custom QLabel to handle mouse clicks on the image area only."""
    left_click_registered = pyqtSignal(QPointF)
    right_click_registered = pyqtSignal(QPointF)
    
    def __init__(self):
        super().__init__()
        self.pixmap = None

    def set_pixmap(self, pixmap):
        self.pixmap = pixmap
        self.setPixmap(pixmap)

    def mousePressEvent(self, event):
        if self.pixmap:
            click_pos = event.pos()
            if event.button() == Qt.LeftButton:
                self.left_click_registered.emit(click_pos)
            elif event.button() == Qt.RightButton:
                self.right_click_registered.emit(click_pos)

    def adjust_pos(self, pos):
        """Adjust the position to the image coordinates."""
        adjusted_x = pos.x() - (self.width() - self.pixmap.width()) / 2
        adjusted_pos = QPointF(adjusted_x, pos.y())
        return adjusted_pos

    def _draw_points(self, labels):
        """Draw a point on the image at the given position."""
        painter = QPainter(self.pixmap)
        painter.setPen(QPen(Qt.red, 5))
        for pos in labels:
            painter.drawPoint(QPointF(pos[0], pos[1]))
        self.setPixmap(self.pixmap)