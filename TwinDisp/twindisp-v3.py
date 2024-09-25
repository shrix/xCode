import cv2
import os
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QSlider, QLabel
from PyQt5.QtCore import Qt, QTimer, QTime, QElapsedTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QKeyEvent, QPalette, QColor

class DualVideoPlayer(QWidget):
    def __init__(self, video1_path, video2_path):
        super().__init__()
        self.video1_path = video1_path
        self.video2_path = video2_path
        print("Initializing video captures...")
        self.cap1 = cv2.VideoCapture(video1_path)
        self.cap2 = cv2.VideoCapture(video2_path)
        
        print("Getting video dimensions...")
        # Get video dimensions
        self.width1 = int(self.cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height1 = int(self.cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width2 = int(self.cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height2 = int(self.cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video 1 dimensions: {self.width1}x{self.height1}")
        print(f"Video 2 dimensions: {self.width2}x{self.height2}")
        
        self.fps1 = self.cap1.get(cv2.CAP_PROP_FPS)
        self.fps2 = self.cap2.get(cv2.CAP_PROP_FPS)
        self.frame_count1 = int(self.cap1.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_count2 = int(self.cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame1 = 0
        self.current_frame2 = 0
        self.playing = False
        self.elapsed_timer = QElapsedTimer()
        self.total_elapsed_time = 0
        print("Calling initUI...")
        self.initUI()
        self.update_button_text()  # Add this line at the end of __init__

    def initUI(self):
        print("Inside initUI...")
        # Set background color for the main window
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(0, 0, 0))  # Black background
        self.setPalette(palette)

        # Calculate window size based on video dimensions with padding
        side_padding = 20
        top_padding = 20
        bottom_padding = 47
        window_width = self.width1 + self.width2 + (3 * side_padding)
        window_height = max(self.height1, self.height2) + top_padding + bottom_padding
        print(f"Setting window size to: {window_width}x{window_height}")
        self.setGeometry(100, 100, window_width, window_height)

        # Create a QFont object for all labels and buttons
        font = QFont("Arial", 12)

        # Set up UI elements
        self.info_label1 = QLabel('0 (0.00)', self)
        self.info_label1.setFont(font)
        self.info_label1.setStyleSheet("color: white;")
        self.info_label1.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.info_label1.setFixedHeight(32)

        self.playBtn = QPushButton('Play (00:00 / 00:00)', self)
        self.playBtn.setFont(font)
        self.playBtn.clicked.connect(self.toggle_play)
        self.playBtn.setFixedHeight(32)

        self.info_label2 = QLabel('0 (0.00)', self)
        self.info_label2.setFont(font)
        self.info_label2.setStyleSheet("color: white;")
        self.info_label2.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.info_label2.setFixedHeight(32)

        # Add slider for timeline
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, self.frame_count1 - 1)
        self.slider.sliderPressed.connect(self.slider_pressed)
        self.slider.sliderReleased.connect(self.slider_released)
        self.slider.valueChanged.connect(self.slider_value_changed)

        # Create layouts
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(side_padding, top_padding, side_padding, 0)

        video_layout = QHBoxLayout()
        video_layout.setSpacing(side_padding)
        self.video_label1 = QLabel(self)
        self.video_label2 = QLabel(self)
        
        # Set size for video labels and center-align them
        self.video_label1.setFixedSize(self.width1, self.height1)
        self.video_label2.setFixedSize(self.width2, self.height2)
        self.video_label1.setAlignment(Qt.AlignCenter)
        self.video_label2.setAlignment(Qt.AlignCenter)

        video_layout.addWidget(self.video_label1)
        video_layout.addWidget(self.video_label2)
        main_layout.addLayout(video_layout)

        # Create a widget for the bottom controls
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 5, 0, 10)
        bottom_layout.setSpacing(2)

        bottom_layout.addWidget(self.slider)

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self.info_label1, 1)
        controls_layout.addWidget(self.playBtn, 7)
        controls_layout.addWidget(self.info_label2, 1)
        
        bottom_layout.addLayout(controls_layout)

        main_layout.addWidget(bottom_widget)

        self.setLayout(main_layout)

        # Setup timer for updating frames
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.setInterval(10)

        # Initial frame display
        self.update_frame()

        # Show the window directly
        self.show()

    def toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            self.elapsed_timer.start()
            self.timer.start()
        else:
            self.total_elapsed_time += self.elapsed_timer.elapsed()
            self.timer.stop()
        self.update_button_text()

    def slider_pressed(self):
        if self.playing:
            self.timer.stop()

    def slider_released(self):
        position = self.slider.value()
        self.set_position(position)
        if self.playing:
            self.elapsed_timer.restart()
            self.timer.start()

    def set_position(self, position):
        self.current_frame1 = position
        self.current_frame2 = int(position * (self.fps2 / self.fps1))
        self.cap1.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame1)
        self.cap2.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame2)
        self.total_elapsed_time = (position / self.fps1) * 1000
        self.elapsed_timer.restart()
        self.update_frame()

    def update_frame(self):
        if self.playing:
            current_time_ms = self.total_elapsed_time + self.elapsed_timer.elapsed()
            self.current_frame1 = int((current_time_ms / 1000) * self.fps1)
            self.current_frame2 = int((current_time_ms / 1000) * self.fps2)

        if self.current_frame1 >= self.frame_count1:
            self.current_frame1 = 0
            self.current_frame2 = 0
            self.total_elapsed_time = 0
            self.elapsed_timer.restart()

        self.slider.setValue(self.current_frame1)

        self.cap1.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame1)
        ret1, frame1 = self.cap1.read()
        if ret1:
            self.display_frame(frame1, self.video_label1)
            self.info_label1.setText(f'{self.current_frame1} ({self.fps1:.2f})')

        self.cap2.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame2)
        ret2, frame2 = self.cap2.read()
        if ret2:
            self.display_frame(frame2, self.video_label2)
            self.info_label2.setText(f'{self.current_frame2} ({self.fps2:.2f})')

        current_time_s = int(self.total_elapsed_time / 1000 + self.elapsed_timer.elapsed() / 1000)
        total_time_s = int(self.frame_count1 / self.fps1)
        
        current_time_str = f"{current_time_s // 60:02d}:{current_time_s % 60:02d}"
        total_time_str = f"{total_time_s // 60:02d}:{total_time_s % 60:02d}"
        
        button_text = 'Pause' if self.playing else 'Play'
        self.playBtn.setText(f'{button_text} ({current_time_str} / {total_time_str})')
        self.update_button_text()

    def display_frame(self, frame, label):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(convert_to_Qt_format))

    def closeEvent(self, event):
        self.cap1.release()
        self.cap2.release()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Space:
            self.toggle_play()
        elif event.key() == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def slider_value_changed(self):
        if not self.playing:
            self.set_position(self.slider.value())

    def set_position(self, position):
        self.current_frame1 = position
        self.current_frame2 = int(position * (self.fps2 / self.fps1))
        self.cap1.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame1)
        self.cap2.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame2)
        self.total_elapsed_time = (position / self.fps1) * 1000
        self.update_frame()

    def update_button_text(self):
        current_time_s = int(self.total_elapsed_time / 1000)
        if self.playing:
            current_time_s += int(self.elapsed_timer.elapsed() / 1000)
        total_time_s = int(self.frame_count1 / self.fps1)
        
        current_time_str = f"{current_time_s // 60:02d}:{current_time_s % 60:02d}"
        total_time_str = f"{total_time_s // 60:02d}:{total_time_s % 60:02d}"
        
        button_text = 'Pause' if self.playing else 'Play'
        self.playBtn.setText(f'{button_text} ({current_time_str} / {total_time_str})')

def main():
    app = QApplication(sys.argv)
    if len(sys.argv) != 3:
        print("Usage: python twindisp.py <video1_path> <video2_path>")
        sys.exit(1)
    
    video1_path = sys.argv[1]
    video2_path = sys.argv[2]
    
    if not os.path.exists(video1_path):
        print(f"Error: Video file '{video1_path}' does not exist.")
        sys.exit(1)
    
    if not os.path.exists(video2_path):
        print(f"Error: Video file '{video2_path}' does not exist.")
        sys.exit(1)
    
    print(f"Video 1 path: {os.path.abspath(video1_path)}")
    print(f"Video 2 path: {os.path.abspath(video2_path)}")
    
    player = DualVideoPlayer(video1_path, video2_path)
    player.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
