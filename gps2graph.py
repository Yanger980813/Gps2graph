'''
Author: Zhengke Sun
Date: 2023/3/18
Contact: zhengkesun@outlook.com
'''
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, \
    QFileDialog, QDesktopWidget, QMessageBox, QTextEdit
from PyQt5 import QtGui
import sys
import datetime
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from scipy.spatial.distance import cdist
from einops import rearrange
from tqdm import tqdm

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Window init
        self.resize(1000, 618)
        self.setWindowTitle('GPS2Graph')
        self.center()

        # Button
        self.button_upload = QPushButton('upload your files', self)
        self.button_upload.setFixedSize(300, 50)
        font1 = QtGui.QFont('Arial', 15)
        self.button_upload.setFont(font1)
        button_width = 300
        button_height = 50
        window_width = self.width()
        window_height = self.height()
        self.button_upload.move(int((window_width - button_width) / 2), int((window_height - button_height) / 2) + 150)

        # Connection
        self.button_upload.clicked.connect(self.upload_file)

        # Text
        self.textbox = QTextEdit(self)
        self.textbox.setAcceptRichText(False)
        self.textbox.setTabChangesFocus(True)
        self.textbox.setUndoRedoEnabled(True)
        self.textbox.setLineWrapMode(QTextEdit.WidgetWidth)

        self.textbox.resize(800, 200)
        self.textbox.setPlaceholderText("Please enter the column titles\n"
                                        "Default: 'id', 'datetime','longitude', 'latitude', 'speed', 'angle', 'status'\n"
                                        "At least: 'id', 'datetime', 'longitude', and 'latitude'")
        self.textbox.setText("'id', 'datetime', 'longitude', 'latitude', 'speed', 'angle', 'status'")
        font2 = QtGui.QFont('Arial', 12)
        self.textbox.setFont(font2)
        text_width = 800
        text_height = 300
        self.textbox.move(int((window_width - text_width) / 2), int((window_height - text_height) / 2) - 100)

    # Move window to center
    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    # Calculate speed(if necessary)
    def cal_speed(self, df):
        time_diff = df['datetime'].diff().dt.total_seconds().fillna(0)
        distance_diff = np.sqrt(((df['latitude'].diff() * 111320) ** 2) + ((df['longitude'].diff() * 111320) ** 2)).fillna(0)
        df['speed'] = distance_diff / time_diff
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        return df

    def upload_file(self):
        file_filter = "CSV files (*.csv);;Text files (*.txt);;All files (**)"
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_names, _ = QFileDialog.getOpenFileNames(self, "choose GPS trajectory files", "",
                                                    file_filter, options=options)

        if file_names:
            # Split text
            input_text = self.textbox.toPlainText()
            input_text = input_text.replace("'", "")
            elements = input_text.split(", ")
            output_list = [element.strip() for element in elements]
            # If data need to calculate speed
            get_speed = False
            if 'speed' not in output_list:
                get_speed = True
            ## Get adjacency matrix
            df = pd.DataFrame()
            for file in file_names:
                df_temp = pd.read_csv(file, names=output_list)
                df_temp['datetime'] = pd.to_datetime(df_temp['datetime'])
                if get_speed == True:
                    df_temp = self.cal_speed(df_temp)
                df = pd.concat([df, df_temp], axis=0)

            # Clustering
            kmeans = KMeans(n_clusters=50)
            kmeans.fit(df[['longitude', 'latitude']])
            centroids = kmeans.cluster_centers_
            df['label'] = kmeans.predict(df[['longitude', 'latitude']])

            # Main nodes
            main_nodes = []
            for i in range(len(centroids)):
                main_nodes.append({'id': i, 'longitude': centroids[i][0], 'latitude': centroids[i][1]})

            # Distance
            dist_matrix = cdist(centroids, centroids, 'euclidean') * 111.32
            threshold = 20.0
            adjacency_matrix = np.zeros((len(centroids), len(centroids)), dtype=int)
            for i in range(len(centroids)):
                for j in range(len(centroids)):
                    if i == j:
                        adjacency_matrix[i, j] = 0
                    elif dist_matrix[i, j] < threshold:
                        adjacency_matrix[i, j] = 1
                    else:
                        adjacency_matrix[i, j] = 0

            # Save adjacency matrix
            adjacency_matrix = pd.DataFrame(adjacency_matrix)
            adjacency_matrix.to_csv('./saved_files/adjacency_matrix.csv', header=False, index=False)

            ## Get feature matrix
            time_interval = datetime.timedelta(minutes=5)
            df = df.sort_values(by=['datetime'])
            groups = df.groupby(pd.Grouper(key='datetime', freq=time_interval))
            speeds = []
            flows = []

            # Nodes
            for i in tqdm(range(len(main_nodes))):
                node_speeds = []
                node_flows = []

                # Time and Labels
                for name, group in groups:
                    group = group[group['label'] == i]
                    count = len(group)
                    if count > 0:
                        speed = group['speed'].mean()
                        node_speeds.append(speed)
                        node_flows.append(count)
                    else:
                        node_speeds.append(0)
                        node_flows.append(0)

                speeds.append(node_speeds)
                flows.append(node_flows)

            speeds = np.array(speeds)
            flows = np.array(flows)

            speeds = rearrange(speeds, 'n t -> t n 1')
            flows = rearrange(flows, 'n t -> t n 1')
            feature_matrix = np.concatenate([speeds, flows], axis=2)
            np.save('./saved_files/feature_matrix.npy', feature_matrix)

            success = True
        else:
            success = False

        if success:

            QMessageBox.information(self,
                                    "Finished",
                                    "The adjacency matrix has been saved to adjacency_matrix.csv, "
                                    "the feature matrix has been saved to feature_matrix.npy",
                                    buttons=QMessageBox.Ok)
        else:

            QMessageBox.information(self,
                                    "Unfinished",
                                    "Please upload files",
                                    buttons=QMessageBox.Ok)

if __name__ == '__main__':

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())