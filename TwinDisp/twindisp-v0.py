import cv2
import os
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QSlider, QLabel
from PyQt5.QtCore import Qt, QTimer, QTime, QElapsedTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QKeyEvent

class DualVideoPlayer(QWidget):
    def __init__(self, video1_path, video2_path):
        super().__init__()
        self.video1_path = video1_path
        self.video2_path = video2_path
        self.cap1 = cv2.VideoCapture(video1_path)
        self.cap2 = cv2.VideoCapture(video2_path)
        self.fps1 = self.cap1.get(cv2.CAP_PROP_FPS)
        self.fps2 = self.cap2.get(cv2.CAP_PROP_FPS)
        self.frame_count1 = int(self.cap1.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_count2 = int(self.cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame1 = 0
        self.current_frame2 = 0
        self.playing = False
        self.elapsed_timer = QElapsedTimer()
        self.total_elapsed_time = 0
        self.initUI()
        self.setFocusPolicy(Qt.StrongFocus)  # Allow the widget to receive keyboard focus

    def initUI(self):
        self.setWindowTitle('Dual Video Player')

        # Retrieve video dimensions
        self.width1 = int(self.cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height1 = int(self.cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width2 = int(self.cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height2 = int(self.cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Set window size based on video dimensions
        window_width = self.width1 + self.width2
        window_height = max(self.height1, self.height2)
        self.setGeometry(100, 100, window_width, window_height)

        # Create video labels
        self.video_label1 = QLabel()
        self.video_label2 = QLabel()

        # Create controls
        self.playBtn = QPushButton('Play')
        self.playBtn.setFont(QFont('Arial', 16))
        self.playBtn.clicked.connect(self.toggle_play)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, self.frame_count1 - 1)
        self.slider.sliderMoved.connect(self.set_position)
        self.slider.sliderPressed.connect(self.slider_pressed)
        self.slider.sliderReleased.connect(self.slider_released)
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.fps_label = QLabel(f"FPS: {self.fps1:.2f} / {self.fps2:.2f}")

        # Create layouts
        main_layout = QVBoxLayout()
        video_layout = QHBoxLayout()
        video_layout.addWidget(self.video_label1)
        video_layout.addWidget(self.video_label2)
        main_layout.addLayout(video_layout)
        main_layout.addWidget(self.slider)
        
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self.fps_label, 1)
        controls_layout.addWidget(self.playBtn, 5)
        controls_layout.addWidget(self.time_label, 1)
        main_layout.addLayout(controls_layout)

        self.setLayout(main_layout)

        # Setup timer for updating frames
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.setInterval(10)  # Update every 10ms

        # Initial frame display
        self.update_frame()

        # Show the window directly
        self.show()

    def toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            self.elapsed_timer.start()
            self.timer.start()
            self.playBtn.setText('Pause')
        else:
            self.total_elapsed_time += self.elapsed_timer.elapsed()
            self.timer.stop()
            self.playBtn.setText('Play')

    def set_position(self, position):
        self.current_frame1 = position
        self.current_frame2 = int(position * (self.fps2 / self.fps1))
        self.cap1.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame1)
        self.cap2.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame2)
        self.update_frame()

    def slider_pressed(self):
        if self.playing:
            self.timer.stop()

    def slider_released(self):
        if self.playing:
            self.elapsed_timer.restart()
            self.timer.start()
        self.update_frame()

    def update_frame(self):
        if self.playing:
            current_time = self.total_elapsed_time + self.elapsed_timer.elapsed()
            self.current_frame1 = int((current_time / 1000) * self.fps1)
            self.current_frame2 = int((current_time / 1000) * self.fps2)

        if self.current_frame1 >= self.frame_count1:
            self.current_frame1 = 0
            self.current_frame2 = 0
            self.elapsed_timer.restart()

        self.cap1.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame1)
        ret1, frame1 = self.cap1.read()
        if ret1:
            self.display_frame(frame1, self.video_label1)
            self.slider.setValue(self.current_frame1)
            self.update_time_label()

        self.cap2.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame2)
        ret2, frame2 = self.cap2.read()
        if ret2:
            self.display_frame(frame2, self.video_label2)

    def display_frame(self, frame, label):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(480, 856, Qt.KeepAspectRatio)
        label.setPixmap(QPixmap.fromImage(p))

    def update_time_label(self):
        current_time = QTime(0, 0).addSecs(int(self.current_frame1 / self.fps1))
        total_time = QTime(0, 0).addSecs(int(self.frame_count1 / self.fps1))
        self.time_label.setText(f"{current_time.toString('mm:ss')} / {total_time.toString('mm:ss')}")

    def closeEvent(self, event):
        self.cap1.release()
        self.cap2.release()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Space:
            self.toggle_play()
        elif event.key() == Qt.Key_Escape:
            self.close()  # This will close the window and quit the application
        else:
            super().keyPressEvent(event)

def main():
    app = QApplication(sys.argv)
    if len(sys.argv) != 3:
        print("Usage: python twindisp.py <video1_path> <video2_path>")
        sys.exit(1)
    
    video1_path = sys.argv[1]
    video2_path = sys.argv[2]
    
    print(f"Video 1 path: {os.path.abspath(video1_path)}")
    print(f"Video 2 path: {os.path.abspath(video2_path)}")
    
    player = DualVideoPlayer(video1_path, video2_path)
    player.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
