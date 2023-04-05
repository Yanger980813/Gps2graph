"""
Author: Zhengke Sun
Update: 2023/4/5
Contact: zhengkesun@outlook.com
"""
from PyQt5.QtWidgets import QApplication, QPushButton, \
    QFileDialog, QDesktopWidget, QMessageBox, QTextEdit, QVBoxLayout, \
    QWidget, QSpacerItem, QSizePolicy, QLabel, QRadioButton, QSpinBox, \
    QDoubleSpinBox, QButtonGroup
from PyQt5 import QtGui
import sys
import pandas as pd
from tqdm import tqdm
from methods.grid import grid
from methods.cluster import cluster
from utils import cal_speed


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Window init
        self.resize(800, 800)
        self.setWindowTitle('GPS2Graph')
        self.center()
        vbox = QVBoxLayout(self)

        # Button
        self.button_upload = QPushButton('upload files', self)
        font1 = QtGui.QFont('Century Gothic', 15)
        self.button_upload.setFont(font1)
        # connection
        self.button_upload.clicked.connect(self.upload_file)

        # Label
        self.label1 = QLabel("Input headers")
        font2 = QtGui.QFont('Courier New', 12)
        font2_1 = QtGui.QFont('Dubai', 14)
        self.label1.setFont(font2_1)
        self.label2 = QLabel("Set parameters and choose transform methods")
        self.label2.setFont(font2_1)
        self.label3 = QLabel("Set feature matrix sampling frequency, default=5")
        self.label3.setFont(font2)
        self.label4 = QLabel("Set cluster nodes, default=50")
        self.label4.setFont(font2)
        self.label5 = QLabel("Set distance threshold, default=5.0")
        self.label5.setFont(font2)
        self.label6 = QLabel("Set io matrix sampling frequency, default=3")
        self.label6.setFont(font2)
        self.label7 = QLabel("Set rows of grid, default=10")
        self.label7.setFont(font2)

        # Text
        self.textbox = QTextEdit(self)
        self.textbox.setAcceptRichText(False)
        self.textbox.setTabChangesFocus(True)
        self.textbox.setUndoRedoEnabled(True)
        self.textbox.setLineWrapMode(QTextEdit.WidgetWidth)
        self.textbox.setPlaceholderText("Default: 'id', 'datetime', 'longitude', 'latitude', "
                                        "'speed', 'angle', 'status'\n"
                                        "At least: 'id', 'datetime', 'longitude', 'latitude'")

        font3 = QtGui.QFont('Times New Roman', 14)
        self.textbox.setFont(font3)

        # Add RadioButton
        self.radio_button1 = QRadioButton("Cluster")
        self.radio_button2 = QRadioButton("Grid")
        font4 = QtGui.QFont('Times New Roman', 13)
        self.radio_button1.setFont(font4)
        self.radio_button2.setFont(font4)
        self.button_group = QButtonGroup()
        self.button_group.addButton(self.radio_button1)
        self.button_group.addButton(self.radio_button2)

        # Add number input module
        self.spinbox1 = QSpinBox(self)
        self.spinbox1.setValue(5)
        self.spinbox2 = QSpinBox(self)
        self.spinbox2.setValue(3)
        self.spinbox3 = QSpinBox(self)
        self.spinbox3.setValue(50)
        self.spinbox4 = QDoubleSpinBox(self)
        self.spinbox4.setValue(5.0)
        self.spinbox5 = QSpinBox(self)
        self.spinbox5.setValue(10)
        # Set layout
        # space
        spacer_item_1 = QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)
        vbox.addItem(spacer_item_1)
        # label1
        vbox.addWidget(self.label1)
        # textbox
        vbox.addWidget(self.textbox, 2)
        # label2
        vbox.addWidget(self.label2)
        # label3/spinbox1
        vbox.addWidget(self.label3)
        vbox.addWidget(self.spinbox1)
        # label6/spinbox2
        vbox.addWidget(self.label6)
        vbox.addWidget(self.spinbox2)
        # radio button 1
        vbox.addWidget(self.radio_button1)
        # label4/spinbox3
        vbox.addWidget(self.label4)
        vbox.addWidget(self.spinbox3)
        # label5/spinbox4
        vbox.addWidget(self.label5)
        vbox.addWidget(self.spinbox4)
        # radio button 2
        vbox.addWidget(self.radio_button2)
        # label7/spinbox5
        vbox.addWidget(self.label7)
        vbox.addWidget(self.spinbox5)
        # space
        spacer_item_2 = QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)
        vbox.addItem(spacer_item_2)
        # upload button
        vbox.addWidget(self.button_upload, 1)
        self.setLayout(vbox)

    # Move window to center
    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def upload_file(self):
        file_filter = "CSV files (*.csv);;Text files (*.txt);;All files (**)"
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_names, _ = QFileDialog.getOpenFileNames(self, "choose GPS trajectory files", "",
                                                     file_filter, options=options)
        method = ''
        if self.button_group.checkedId() != -1:
            selected_button = self.button_group.checkedButton()
            method = selected_button.text()

        if file_names:
            # Split text
            input_text = self.textbox.toPlainText()
            if input_text == '':
                input_text = "'id', 'datetime', 'longitude', 'latitude', 'speed', 'angle', 'status'"
            input_text = input_text.replace("'", "")
            elements = input_text.split(", ")
            output_list = [element.strip() for element in elements]
            # If data need to calculate speed
            get_speed = False
            if 'speed' not in output_list:
                get_speed = True
            # Get full data
            df = pd.DataFrame()
            print('Processing data')
            for file in tqdm(file_names):
                df_temp = pd.read_csv(file, names=output_list)
                df_temp['datetime'] = pd.to_datetime(df_temp['datetime'])
                if get_speed is True:
                    df_temp = cal_speed(df_temp)
                df = pd.concat([df, df_temp], axis=0)

            # Clean data
            df.dropna(inplace=True)
            df = df[(df['speed'] >= 0) & (df['speed'] <= 150)]
            df = df[
                (df['longitude'] >= -180) & (df['longitude'] <= 180) & (df['latitude'] >= -90) & (df['latitude'] <= 90)]
            df.drop_duplicates(subset=['id', 'datetime'], keep='first', inplace=True)
            df = df.sort_values(by=['id', 'datetime'])
            # Drop speed error data
            speed_error_mask = (df['id'] == df['id'].shift(1)) & (df['longitude'] == df['longitude'].shift(1)) & \
                               (df['latitude'] == df['latitude'].shift(1)) & (df['speed'] != 0)
            df = df[~speed_error_mask]
            df.reset_index(drop=True, inplace=True)

            # [Start]select the methods
            fm_interval = self.spinbox1.value()
            io_interval = self.spinbox2.value()
            nodes_num = self.spinbox3.value()
            threshold = self.spinbox4.value()
            blocks_num = self.spinbox5.value()
            if method == 'Grid':
                grid(df, blocks_num=blocks_num, fm_interval=fm_interval, io_interval=io_interval)
            elif method == 'Cluster':
                cluster(df, nodes_num=nodes_num, fm_interval=fm_interval, io_interval=io_interval, threshold=threshold)
            # [End]select the methods

            success = True
        else:
            success = False

        if success:
            QMessageBox.information(self,
                                    "Finished",
                                    "The results have been saved!",
                                    buttons=QMessageBox.Ok)
        else:
            QMessageBox.information(self,
                                    "Unfinished",
                                    "Please upload files!",
                                    buttons=QMessageBox.Ok)


if __name__ == '__main__':

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())