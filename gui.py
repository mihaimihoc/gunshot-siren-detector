import sys
import detector
import time
import threading
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QFrame, QTextEdit, QPushButton
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QColor, QPalette


class SignalBridge(QObject): # QObject to bridge signals from detector thread to GUI thread
    new_result = pyqtSignal(dict)

bridge = SignalBridge()


class ConfidenceBar(QWidget):
    def __init__(self, label, color, parent=None):
        super().__init__(parent)
        self.color = color

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Label + percentage row
        top_row = QHBoxLayout()
        self.name_label = QLabel(label)
        self.name_label.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.name_label.setStyleSheet("color: #e0e0e0;")

        self.pct_label = QLabel("0%")
        self.pct_label.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.pct_label.setStyleSheet(f"color: {color};")
        self.pct_label.setAlignment(Qt.AlignRight)

        top_row.addWidget(self.name_label)
        top_row.addWidget(self.pct_label)
        layout.addLayout(top_row)

        # Bar background
        self.bar_bg = QFrame()
        self.bar_bg.setFixedHeight(22)
        self.bar_bg.setStyleSheet("background-color: #2a2a3a; border-radius: 11px;")
        self.bar_bg.setLayout(QHBoxLayout())
        self.bar_bg.layout().setContentsMargins(0, 0, 0, 0)

        # Bar fill
        self.bar_fill = QFrame(self.bar_bg)
        self.bar_fill.setFixedHeight(22)
        self.bar_fill.setStyleSheet(
            f"background-color: {color}; border-radius: 11px;"
        )
        self.bar_fill.setFixedWidth(0)

        layout.addWidget(self.bar_bg)

    def set_value(self, value: float):
        pct = int(value * 100)
        self.pct_label.setText(f"{pct}%")
        max_width = self.bar_bg.width() or 400
        self.bar_fill.setFixedWidth(int(max_width * value))


class AlertPanel(QLabel):
    def __init__(self):
        super().__init__("🎙️  LISTENING")
        self.setAlignment(Qt.AlignCenter)
        self.setFont(QFont("Segoe UI", 26, QFont.Bold))
        self.setFixedHeight(100)
        self.set_idle()

    def set_idle(self):
        self.setText("🎙️  LISTENING")
        self.setStyleSheet("""
            background-color: #1e1e2e;
            color: #555577;
            border-radius: 12px;
            border: 2px solid #333355;
        """)

    def set_alert(self, label, confidence):
        icon = "🔫" if label == "Gunshot" else "🚨"
        self.setText(f"{icon}  {label.upper()} DETECTED  —  {int(confidence*100)}%")
        color = "#ff4444" if label == "Gunshot" else "#ff9900"
        self.setStyleSheet(f"""
            background-color: {color}22;
            color: {color};
            border-radius: 12px;
            border: 2px solid {color};
        """)


class MainWindow(QMainWindow):
    CONFIDENCE_THRESHOLD = 0.6
    ALERT_HOLD_SECONDS   = 2.0  

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sound Detector — Gunshot & Siren")
        self.setMinimumSize(700, 580)
        self._last_alert_time = 0
        self._build_ui()

        bridge.new_result.connect(self._on_result)

    def _build_ui(self):
        self.setStyleSheet("background-color: #12121e; color: #e0e0e0;")

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(20)

        # Title
        title = QLabel("🔊 Sound Detector")
        title.setFont(QFont("Segoe UI", 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #aaaaff;")
        main_layout.addWidget(title)

        # Alert panel
        self.alert_panel = AlertPanel()
        main_layout.addWidget(self.alert_panel)

        # Confidence bars
        bars_frame = QFrame()
        bars_frame.setStyleSheet(
            "background-color: #1a1a2e; border-radius: 12px; padding: 10px;"
        )
        bars_layout = QVBoxLayout(bars_frame)
        bars_layout.setSpacing(14)

        bars_title = QLabel("Live Confidence")
        bars_title.setFont(QFont("Segoe UI", 10))
        bars_title.setStyleSheet("color: #666688;")
        bars_layout.addWidget(bars_title)

        self.bar_gunshot = ConfidenceBar("Gunshot",    "#ff4444")
        self.bar_siren   = ConfidenceBar("Siren",      "#ff9900")
        self.bar_bg      = ConfidenceBar("Background", "#44bb88")

        for bar in [self.bar_gunshot, self.bar_siren, self.bar_bg]:
            bars_layout.addWidget(bar)

        main_layout.addWidget(bars_frame)

        # Log
        log_label = QLabel("Detection Log")
        log_label.setFont(QFont("Segoe UI", 10))
        log_label.setStyleSheet("color: #666688;")
        main_layout.addWidget(log_label)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(160)
        self.log_box.setFont(QFont("Courier New", 9))
        self.log_box.setStyleSheet("""
            background-color: #0e0e1a;
            color: #99bbff;
            border: 1px solid #333355;
            border-radius: 8px;
            padding: 6px;
        """)
        main_layout.addWidget(self.log_box)

        # Clear button
        clear_btn = QPushButton("Clear Log")
        clear_btn.setFixedWidth(100)
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a2a4a;
                color: #aaaacc;
                border: 1px solid #444466;
                border-radius: 6px;
                padding: 5px 12px;
            }
            QPushButton:hover { background-color: #3a3a5a; }
        """)
        clear_btn.clicked.connect(self.log_box.clear)
        main_layout.addWidget(clear_btn, alignment=Qt.AlignRight)

        # Timer to reset alert panel
        self._alert_timer = QTimer()
        self._alert_timer.timeout.connect(self._check_alert_expiry)
        self._alert_timer.start(200)

    def _on_result(self, result: dict):
        # Update bars
        self.bar_gunshot.set_value(result["gunshot"])
        self.bar_siren.set_value(result["siren"])
        self.bar_bg.set_value(result["background"])

        # Alert panel & logging
        confirmed = result.get("alert")
        if confirmed:
            conf = result["gunshot"] if confirmed == "Gunshot" else result["siren"]
            
            # Update panel
            self.alert_panel.set_alert(confirmed, conf)
            self._last_alert_time = time.time()

            # Add to log
            ts = time.strftime("%H:%M:%S")
            self.log_box.append(f"[{ts}]  🚨  {confirmed} {int(conf*100)}%")
            self.log_box.verticalScrollBar().setValue(
                self.log_box.verticalScrollBar().maximum()
            )

    def _check_alert_expiry(self):
        if (self.alert_panel.text() != "🎙️  LISTENING" and
                time.time() - self._last_alert_time > self.ALERT_HOLD_SECONDS):
            self.alert_panel.set_idle()


def main():
    # Start detector in background thread
    def detection_loop():
        def on_result(result):
            bridge.new_result.emit(result)
        detector.start_stream(on_result)

    thread = threading.Thread(target=detection_loop, daemon=True)
    thread.start()

    # Launch GUI
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()