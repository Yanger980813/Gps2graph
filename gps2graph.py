"""
Author: Zhengke Sun
Update: 2023/3/29
Contact: zhengkesun@outlook.com
"""
from PyQt5.QtWidgets import QApplication, QPushButton, \
    QFileDialog, QDesktopWidget, QMessageBox, QTextEdit, QVBoxLayout, \
    QWidget, QSpacerItem, QSizePolicy
from PyQt5 import QtGui
import sys
import pandas as pd
from tqdm import tqdm
from methods.gridio import grid_io
from utils import cal_speed


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Window init
        self.resize(1000, 618)
        self.setWindowTitle('GPS2Graph')
        self.center()
        vbox = QVBoxLayout(self)

        # Button
        self.button_upload = QPushButton('upload files', self)
        font1 = QtGui.QFont('Segoe Print', 15)
        self.button_upload.setFont(font1)

        # Connection
        self.button_upload.clicked.connect(self.upload_file)

        # Text
        self.textbox = QTextEdit(self)
        self.textbox.setAcceptRichText(False)
        self.textbox.setTabChangesFocus(True)
        self.textbox.setUndoRedoEnabled(True)
        self.textbox.setLineWrapMode(QTextEdit.WidgetWidth)
        self.textbox.setPlaceholderText("Enter column titles, the side length number of grid cells and io time "
                                        "interval\n"
                                        "\n"
                                        "Default: 'id', 'datetime', 'longitude', 'latitude', "
                                        "'speed', 'angle', 'status', '10', '3'\n"
                                        "At least: 'id', 'datetime', 'longitude', 'latitude', '10', '3'")

        font2 = QtGui.QFont('Times New Roman', 14)
        self.textbox.setFont(font2)

        # Set layout
        spacer_item_1 = QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)
        vbox.addItem(spacer_item_1)
        vbox.addWidget(self.textbox, 2)
        spacer_item_2 = QSpacerItem(20, 250, QSizePolicy.Minimum, QSizePolicy.Expanding)
        vbox.addItem(spacer_item_2)
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

        if file_names:
            # Split text
            input_text = self.textbox.toPlainText()
            if input_text == '':
                input_text = "'id', 'datetime', 'longitude', 'latitude', 'speed', 'angle', 'status', '10', '3'"
            input_text = input_text.replace("'", "")
            elements = input_text.split(", ")
            output_list = [element.strip() for element in elements]
            blocks_num = int(output_list[-2])
            io_interval = int(output_list[-1])
            output_list = output_list[0:-2]
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

            # [Start]select the methods
            grid_io(df, blocks_num=blocks_num, io_interval=io_interval)
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
