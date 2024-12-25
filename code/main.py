from sys import exit, argv
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QPushButton, QFileDialog, QMainWindow, QAction, QToolBar, \
    QScrollArea, QDialog, QHBoxLayout, QDoubleSpinBox, QTableWidget, QTabWidget,QWidget
from PyQt5.QtCore import Qt, QEventLoop
from PyQt5.QtGui import QImage, QPixmap
from tranform import *
from os import path, listdir

class GenericParameterDialog(QDialog):
    def __init__(self, function_name, parameter_names, default_values=None):
        super().__init__()

        self.function_name = function_name
        self.parameter_names = parameter_names
        self.previous_values = default_values or {}
        self.parameter_inputs = []
        # 初始化UI
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(f'{self.function_name} Parameters')

        # 创建输入参数的标签和文本框
        for parameter_name in self.parameter_names:
            label = QLabel(f'{parameter_name}:')
            input_field = QDoubleSpinBox()
            input_field.setMaximum(float('inf'))
            input_field.setSingleStep(5.0)
            input_field.setValue(float(self.previous_values.get(parameter_name, 0.0)))
            self.parameter_inputs.append(input_field)

        # 创建按钮
        ok_button = QPushButton('OK')
        ok_button.clicked.connect(self.accept)

        cancel_button = QPushButton('Cancel')
        cancel_button.clicked.connect(self.reject)

        # 创建布局
        layout = QVBoxLayout()

        for label, input_field in zip(self.parameter_names, self.parameter_inputs):
            parameter_layout = QHBoxLayout()
            parameter_layout.addWidget(QLabel(f'{label}:'))
            parameter_layout.addWidget(input_field)
            layout.addLayout(parameter_layout)

        button_layout = QHBoxLayout()
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def get_parameters(self):
        # 获取输入的参数
        parameters = [input_field.value() for input_field in self.parameter_inputs]
        return tuple(parameters)


class ConvolutionKernelDialog(QDialog):
    def __init__(self, rows, columns, default_kernel=None):
        super().__init__()

        self.function_name = 'Convolution Kernel'
        self.kernel_size = (rows, columns)
        self.kernel_input = [[QDoubleSpinBox() for _ in range(columns)] for _ in range(rows)]

        # 如果提供了默认卷积核，将其设置到输入框中
        if default_kernel is not None:
            for i in range(rows):
                for j in range(columns):
                    self.kernel_input[i][j].setValue(default_kernel[i, j])

        # 初始化UI
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(f'{self.function_name} Parameters')

        # 创建表格布局
        table_layout = QTableWidget()
        table_layout.setRowCount(self.kernel_size[0])
        table_layout.setColumnCount(self.kernel_size[1])

        # 将QDoubleSpinBox放入表格中
        for i in range(self.kernel_size[0]):
            for j in range(self.kernel_size[1]):
                # 创建 QDoubleSpinBox，设置最小值为负无穷，最大值为正无穷
                self.kernel_input[i][j].setMinimum(float('-inf'))
                self.kernel_input[i][j].setMaximum(float('inf'))
                self.kernel_input[i][j].setValue(float(0.5 * (-1) ** j))

                # 将 QDoubleSpinBox 放入表格
                table_layout.setCellWidget(i, j, self.kernel_input[i][j])

        # 创建按钮
        ok_button = QPushButton('OK')
        ok_button.clicked.connect(self.accept)

        cancel_button = QPushButton('Cancel')
        cancel_button.clicked.connect(self.reject)

        # 创建布局
        layout = QVBoxLayout()
        layout.addWidget(table_layout)

        button_layout = QHBoxLayout()
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def get_parameters(self):
        # 获取输入的卷积核
        kernel = np.zeros(self.kernel_size, dtype=float)

        for i in range(self.kernel_size[0]):
            for j in range(self.kernel_size[1]):
                kernel[i, j] = self.kernel_input[i][j].value()

        return kernel

class RibbonTab(QWidget):
    def __init__(self, title, actions):
        super().__init__()
        layout = QVBoxLayout()

        toolbar = QToolBar()

        for label, method in actions.items():
            action = QAction(label, self)
            action.triggered.connect(method)
            toolbar.addAction(action)

        layout.addWidget(toolbar)
        self.setLayout(layout)
        self.tab_title = title


class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        # 初始化UI
        self.init_ui()
        # 创建文件对话框实例，以备将来使用
        self.file_dialog = QFileDialog()
        self.resize(1000, 1200)
        self.event_loop = None
        # 创建字典用于保存不同类型窗口的上次输入值
        self.previous_values_by_type = {
            'Circle Recognition': {'param1': 45, 'param2': 30, 'minDist': 100, 'minRadius': 50, 'maxRadius': 250},
            'Line Recognition': {'threshold': 90, 'minLineLength': 150, 'maxLineGap': 100},
            'convolution': {'Kernel Rows': 1, 'Kernel Columns': 3, 'Threshold': 127},
            'Threshold Image': {'threshold': 127}
        }
        # 初始化图像变量
        self.current_image = ImageProcessor()
        self.show_image = np.array([])
        self.file_path = str()
        self.source_folder = str()
        self.begin = False
        self.recorded_operations_temp = []
        self.recorded_operations = []
        self.clicked_coordinate = (np.inf, np.inf)
        self.coordinate_picker_enabled = False
        self.tempfeaturesmanager = FeatureManager()

    def init_ui(self):
        # 设置主窗口的标题
        self.setWindowTitle('Image Viewer')

        self.ribbon_tabs = QTabWidget(self)
        self.setMenuWidget(self.ribbon_tabs)

        # 创建一个滚动区域
        scroll_area = QScrollArea(self)
        self.setCentralWidget(scroll_area)

        # 创建一个标签用于显示图片
        self.image_label = QLabel(scroll_area)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll_area.setWidget(self.image_label)

        # 设置滚动区域的属性
        scroll_area.setWidgetResizable(True)

        # 添加功能区选项卡
        file_tab = RibbonTab("File", {
             "Open": self.show_dialog,
             "Save": self.save_image,
             # "open folder": self.process_image
        })

        process_tab = RibbonTab("Process", {
            "save path": self.save_path,
            "Convolution": self.show_generic_parameter_dialog_convolution,
            "Threshold Image": self.show_generic_parameter_dialog_threshold_image,
            "Negative Image": self.show_negative_image,
            "Blend": self.blend,
            "Different Checker": self.different_checker
        })

        features_tab = RibbonTab("Features", {
            "Choose": self.get_clicked_coordinate,
            "save features": self.save_features,
            "Circle": self.show_generic_parameter_dialog_circle,
            "Line": self.show_generic_parameter_dialog_line,
            "Contour": self.show_generic_parameter_dialog_contour,

        })

        batch_tab = RibbonTab("Batch Process",{
            "begin": self.batch_process_begin,
            "end": self.batch_process_end,
            "batch process": self.batch_process_folder
        })

        self.ribbon_tabs.addTab(file_tab, "File")
        self.ribbon_tabs.addTab(process_tab, "Process")
        self.ribbon_tabs.addTab(features_tab, "Features")
        self.ribbon_tabs.addTab(batch_tab, "Batch Process")

        # 在图像标签上添加点击事件处理程序
        self.image_label.mousePressEvent = self.mouse_click_event



    def show_dialog(self):
        # 弹出文件选择对话框
        self.file_path, _ = self.file_dialog.getOpenFileName(self, 'Open Image File', '',
                                                             'Images (*.png *.jpg *.jpeg *.bmp *.tif)')

        if self.file_path:
            # 读取图像
            self.current_image = ImageProcessor(imread(self.file_path))
            self.show_image = ImageProcessor(imread(self.file_path)).image

            # 将图像转换为 QPixmap 并显示
            pixmap = self.to_pixmap()
            self.image_label.setPixmap(pixmap)
            self.image_label.show()

    def save_image(self):
        if self.current_image is not None:
            # 弹出保存文件对话框
            save_path, _ = self.file_dialog.getSaveFileName(self, 'Save Image File', '',
                                                            'Images (*.jpg *.png *.jpeg *.bmp)')

            if save_path:
                # 保存当前图像
                imwrite(save_path, self.current_image.image)

    def mouse_click_event(self, ev):
        if self.coordinate_picker_enabled and self.current_image is not None:
            # 获取点击坐标
            x = int(ev.pos().x())
            y = int(ev.pos().y())

            # 保存点击的坐标
            self.clicked_coordinate = (x, y)

            # 取消坐标获取器的启用状态
            self.coordinate_picker_enabled = False

            if self.event_loop is not None and self.event_loop.isRunning():
                self.event_loop.exit()

    def get_clicked_coordinate(self):
        self.coordinate_picker_enabled = True

        # 清空之前保存的坐标
        self.clicked_coordinate = (np.inf, np.inf)

        # 创建一个新的事件循环
        self.event_loop = QEventLoop()

        # 运行事件循环以等待鼠标点击事件
        self.event_loop.exec_()
        print(self.clicked_coordinate)
        nearest_feature = self.current_image.features_manager.get_nearest_feature((self.clicked_coordinate))
        
        print(nearest_feature)
        return self.clicked_coordinate

    def to_pixmap(self):
        # 将当前图像转换为 QPixmap 对象，以便在 Qt 框架中显示
        show_image = np.array(self.show_image)
        image_type = ImageProcessor(show_image).type
        if image_type == "Grayscale":
            # 如果是灰度图，使用 QImage 格式将其转换为 QPixmap
            q_image = QImage(show_image.data, show_image.shape[1], show_image.shape[0],
                             show_image.strides[0], QImage.Format_Grayscale8)
        elif image_type == "RGB":
            # 如果是 RGB 彩色图像，先转换为 RGB 格式
            q_image = QImage(show_image.data, show_image.shape[1], show_image.shape[0],
                             show_image.strides[0], QImage.Format_RGB888)
        else:
            # 如果是未知类型的图像，设置为无效格式
            q_image = QImage(show_image.data, show_image.shape[1], show_image.shape[0],
                             show_image.strides[0], QImage.Format_Invalid)
            
        return QPixmap.fromImage(q_image)  # 返回对应的 QPixmap 对象

    def save_path(self):
        if self.show_image is not None:
            self.current_image.update_image(self.show_image)
        if self.begin:
            self.recorded_operations.append(self.recorded_operations_temp[-1])

    def save_features(self):
        if self.current_image.features_manager.features is not None:
            self.current_image.features_manager.confirmed()
            self.save_choose_feature()
        if self.begin:
            self.recorded_operations.append(self.recorded_operations_temp[-1])

    def save_choose_feature(self):
        if self.current_image.features_manager.features is not None:
            nearest_feature = self.current_image.features_manager.get_nearest_feature((self.clicked_coordinate))
            FeatureManager([nearest_feature]).save_to_csv("example_output_with_header.csv")

    def batch_process_begin(self):
        self.begin = True
        self.recorded_operations_temp = []

    def batch_process_end(self):
        self.begin = False
        print(self.recorded_operations)

    def show_generic_parameter_dialog(self, function_name, parameter_names):
        if parameter_names is not None:
            # 弹出通用参数输入对话框
            dialog = GenericParameterDialog(function_name, parameter_names,
                                            self.previous_values_by_type.get(function_name, {}))
            result = dialog.exec_()
            if result == QDialog.Accepted:
                # 用户按下 OK 按钮，获取参数并执行相应的操作
                parameters = dialog.get_parameters()
                parameters_dict = dict(zip(parameter_names, parameters))
                print(f"{function_name} parameters: {parameters_dict}")
                # 保存当前窗口类型的上次输入值
                self.previous_values_by_type[function_name] = parameters_dict
            else:
                # 用户按下 Cancel 按钮，返回空字典或者其他适当的数值
                return 'cancel'
        else:
            parameters_dict = None
        return parameters_dict

    def generic_image_command(self, processing_method, function_name=None, parameter_names=None):
        # 显示通用参数输入对话框
        parameters_dict_self = self.show_generic_parameter_dialog(function_name, parameter_names)
        if parameters_dict_self != 'cancel':
            if self.begin:
                operation = {'type': 'image_processing', 'function_name': function_name,
                             'processing_method': processing_method, 'parameters': parameters_dict_self}
                self.recorded_operations_temp.append(operation)
            if self.current_image.image is not None:
                if parameter_names is not None:
                    processed_image = processing_method(self.current_image, parameters_dict_self)
                else:
                    processed_image = processing_method(self.current_image)
                if processed_image is not None:
                    self.show_image = processed_image
                    pixmap = self.to_pixmap()
                    self.image_label.setPixmap(pixmap)
                    self.image_label.show()

    def show_generic_parameter_dialog_threshold_image(self):
        function_name = 'Threshold Image'
        parameter_names = ['threshold', 'method']
        self.generic_image_command(ImageProcessor.threshold_image, function_name, parameter_names)

    def blend(self):
        new_file_path, _ = self.file_dialog.getOpenFileName(self, 'Open Image File', '',
                                                            'Images (*.png *.jpg *.jpeg *.bmp *.tif)')

        if new_file_path:
            # Read the new image
            new_image = imread(new_file_path)

            if self.current_image.image is not None:
                blended_image = self.current_image.negative_blend(new_image)

                # Update the image_label with the blended image
                if blended_image is not None:
                    self.show_image = blended_image
                    pixmap = self.to_pixmap()
                    self.image_label.setPixmap(pixmap)
                    self.image_label.show()

    def different_checker(self):
        new_file_path, _ = self.file_dialog.getOpenFileName(self, 'Open Image File', '',
                                                            'Images (*.png *.jpg *.jpeg *.bmp *.tif)')

        if new_file_path:
            # Read the new image
            new_image = imread(new_file_path)

            if self.current_image.image is not None:
                blended_image = self.current_image.diffcheck(new_image)

                # Update the image_label with the blended image
                if blended_image is not None:
                    self.show_image = blended_image
                    pixmap = self.to_pixmap()
                    self.image_label.setPixmap(pixmap)
                    self.image_label.show()

    def show_generic_parameter_dialog_circle(self):
        function_name = 'Circle Recognition'
        parameter_names = ['param1', 'param2', 'minDist', 'minRadius', 'maxRadius']
        self.generic_image_command(ImageProcessor.circle_recongnition, function_name, parameter_names)

    def show_generic_parameter_dialog_line(self):
        function_name = 'Line Recognition'
        parameter_names = ['threshold', 'minLineLength', 'maxLineGap']
        self.generic_image_command(ImageProcessor.line_recongition, function_name, parameter_names)

    def show_generic_parameter_dialog_contour(self):
        function_name = 'Contour Recognition'
        parameter_names = None
        self.generic_image_command(ImageProcessor.contour_recongition, function_name, parameter_names)

    def show_negative_image(self):
        function_name = 'Negative Image'
        parameter_names = None
        self.generic_image_command(ImageProcessor.negative_image, function_name, parameter_names)

    def show_generic_parameter_dialog_convolution(self):
        # 显示通用参数输入对话框（用于卷积核定义）
        if self.current_image.image is not None:
            function_name = 'convolution'
            parameter_names = ['Kernel Rows', 'Kernel Columns', 'Threshold']
            convolution_parameters_dict = self.show_generic_parameter_dialog(function_name, parameter_names)
            if convolution_parameters_dict != 'cancel' and convolution_parameters_dict is not None:
                rows, columns, threshold = (int(convolution_parameters_dict['Kernel Rows']),
                                             int(convolution_parameters_dict['Kernel Columns']),
                                             int(convolution_parameters_dict['Threshold']))
                if rows is not None and columns is not None:
                    kernel_dialog = ConvolutionKernelDialog(rows, columns)
                    result = kernel_dialog.exec_()
                    if result == QDialog.Accepted:
                        # 用户按下 OK 按钮，获取卷积核并执行相应的操作
                        kernel = kernel_dialog.get_parameters()
                        print(f"Convolution kernel: {kernel}")
                        convolution_parameters_dict = {'kernel': kernel, 'threshold': threshold}
                        # 保存当前窗口类型的上次输入值
                        self.previous_values_by_type['Convolution'] = {'Kernel Rows': rows,
                                                                       'Kernel Columns': columns,
                                                                       'Threshold': threshold}
                        current_image = self.current_image
                        convolution_image = self.current_image.convolution(convolution_parameters_dict)
                        if self.begin:
                            operation = {'type': 'image_processing', 'function_name': function_name,
                                         'parameters': convolution_parameters_dict}
                            self.recorded_operations_temp.append(operation)
                        self.show_image = convolution_image
                        pixmap = self.to_pixmap()
                        self.image_label.setPixmap(pixmap)
                        self.image_label.show()
        else:
            print("no image")
            return

    def batch_process_folder(self):
        self.source_folder = QFileDialog.getExistingDirectory(self, 'Select Source Folder', self.source_folder)
        if not self.source_folder:
            print("Source folder not selected.")
            return None

        # 弹出文件夹选择对话框，获取目标文件夹
        target_folder = QFileDialog.getExistingDirectory(self, 'Select Target Folder', self.source_folder)
        if not target_folder:
            print("Target folder not selected.")
            return None

        for filename in listdir(self.source_folder):
            if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_path = path.join(self.source_folder, filename)
                self.batch_process_image(image_path, target_folder)

    def batch_process_image(self, image_path: str, target_folder: str):
        # 读取图像
        print(image_path)
        image = imread(image_path)

        # 应用记录的操作
        for operation in self.recorded_operations:
            if operation['type'] == 'image_processing':
                function_name = operation['function_name']
                processing_method = operation['processing_method']
                parameters = operation['parameters']

                # 执行操作
                image = ImageProcessor(image)
                image = processing_method(image, parameters)

        # 显示处理后的图像
        processed_image = ImageProcessor(image)
        self.show_image = processed_image

        # 保存处理后的图像到目标文件夹
        base_filename, _ = path.splitext(path.basename(image_path))
        save_path = path.join(target_folder, f"{base_filename}_processed.png")
        imwrite(save_path, processed_image.image)

        # 更新界面显示
        pixmap = self.to_pixmap()
        self.image_label.setPixmap(pixmap)
        self.image_label.show()


if __name__ == '__main__':
    app = QApplication(argv)
    viewer = ImageViewer()
    viewer.show()
    exit(app.exec_())
