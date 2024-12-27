import cv2
import numpy as np
from PIL import Image


class ImageProcessor:
    def __init__(self, image: np.ndarray = np.array([])):  # 初始化，接受图像作为输入，初始化图像类型和特征管理器
        super().__init__()
        self.image = image  # 设置图像
        self.type = self.detect_type()   # 自动检测图像类型
        self.features_manager = FeatureManager([])  # 初始化特征管理器
        self.temp_features = FeatureManager([])     # 临时特征管理器

    def detect_type(self) -> str:
        """检测图像类型，返回图像类型的字符串"""
        if len(self.image.shape) == 2:
            return "Grayscale"      # 如果是二维数组，表示灰度图
        elif len(self.image.shape) == 3 and self.image.shape[2] == 3:
            return "RGB"            # 如果是三维数组且最后一维为3，表示RGB彩色图
        else:
            return "Empty" # 其他类型的图像无法处理

    def update_image(self, new_image) -> 'ImageProcessor':
        # 更新当前图像并重新检测图像类型
        self.image = new_image  # 设置新的图像
        self.type = self.detect_type()  # 自动检测新图像的类型
        return self  # 返回当前对象实例，便于链式调用



    def to_rgb(self, method='weighted') -> np.ndarray:

        """
        将图像转换为RGB格式，支持多种转换方案

        - method (str): 转换方法，可选值包括：
            - 'weighted'  : 灰度图像加权填充到RGB通道
            - 'colormap'  : 使用彩色映射将灰度图转换为RGB

        :param method: 转换方案
        :return: 返回RGB格式的图像
        """

        if self.type == "RGB":
            return self.image
        elif self.type == "Grayscale":
            # 如果是灰度图像，直接处理
            gray_image = self.image

            if method == 'weighted':
                # 灰度图像加权填充到RGB
                # 假设灰度图像已是二维矩阵，直接复制到RGB的三个通道
                rgb_image = np.stack([gray_image] * 3, axis=-1)  # (H, W) -> (H, W, 3)
            elif method == 'colormap':
                # 使用彩色映射将灰度图转换为RGB
                # 这里我们使用一个简单的线性映射，但可以根据需要更改为其他映射方法
                colormap = np.array([
                    [0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255],  # 红绿蓝
                    [255, 255, 0], [0, 255, 255], [255, 0, 255], [255, 255, 255]  # 黄色、青色、紫色、白色
                ])
                # 线性映射灰度图像为RGB
                norm_gray = np.clip(gray_image, 0, 255)  # 确保灰度值在0到255范围
                indices = (norm_gray // 32).astype(int)  # 假设分为8个区间
                rgb_image = np.array(colormap[indices])  # 映射到RGB颜色
            else:
                raise ValueError(f"Unsupported method: {method}")  # 不支持的转换方法
            return rgb_image

        else:
            raise TypeError(f"Unsupported method: {method} for image type: {self.type}")    # 不支持的转换方法

    def to_grayscale(self, method='weighted')   -> np.ndarray:
        """
        将图像转换为灰度图，支持多种灰度化方案

        -method(str) : 灰度化方法，可选值包括：
            - 'average'  : 平均法 (取RGB通道的均值)
            - 'weighted' : 加权平均法 (常用的加权方法)
            - 'ntsc'     : NTSC标准灰度化
            - 'max'      : 最大值法 (取RGB通道中的最大值)
            - 'min'      : 最小值法 (取RGB通道中的最小值)
        :param method:灰度化方法
        :return: 返回灰度化后的图像
        """
        if self.type == "Grayscale":
            # 如果已经是灰度图，直接返回
            return self.image.copy()

        elif self.type == "RGB":

            if method == 'average':
                # 平均法：取RGB的平均值
                gray_image = self.image.mean(axis=2).astype(np.uint8)

            elif method == 'weighted':
                # 加权平均法：常用的加权方法
                gray_image = np.dot(self.image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

            elif method == 'ntsc':
                # NTSC标准：使用公式 R * 0.2989 + G * 0.5870 + B * 0.1140
                gray_image = np.dot(self.image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

            elif method == 'max':
                # 最大值法：取RGB中最大值作为灰度值
                gray_image = np.max(self.image, axis=2).astype(np.uint8)

            elif method == 'min':
                # 最小值法：取RGB中最小值作为灰度值
                gray_image = np.min(self.image, axis=2).astype(np.uint8)

            else:
                # 默认使用加权平均法
                gray_image = np.dot(self.image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

            # 返回灰度图像
            return gray_image

        else:
            raise TypeError(f"Unsupported method: {method} for image type: {self.type}")
        
    def negative_image(self)    -> np.ndarray:
        """创建图像的负片效果，返回反转后的图像"""
        negative_image = 255 - self.image  # 对每个通道取反
        self.image = negative_image
        return negative_image  # 返回负片图像

    def threshold_image(self, threshold_parameters_dict:dict)    -> np.ndarray:
        """
        将图像转换为二值图像，并根据给定的阈值处理

        - threshold_parameters_dict (dict): 包含阈值化所需的参数，具体字段如下：
          - 'method' (int): 选择阈值化的方法（可选值：0-'standard', 1-'adaptive', 2-'otsu'）
          - 'threshold' (int): 对于标准阈值化，指定固定阈值
          - 'block_size' (int): 对于自适应阈值化，指定局部区域大小（奇数）
          - 'c' (int): 对于自适应阈值化，指定常数值，用于调整阈值
        :param threshold_parameters_dict: 参数字典，结构为：{'threshold'：threshold}
        :return: 返回阈值化后的二值图像
        """
        image = self.to_grayscale()  # 转换为灰度图像
        method = threshold_parameters_dict.get('method', 'standard')

        # 处理不同的阈值化方法
        if method == 'standard' or method == 0.0:
            # 使用固定阈值进行二值化
            threshold_value = threshold_parameters_dict.get('threshold', 127)  # 默认阈值为127
            binary_image = np.where(image > threshold_value, 255, 0).astype(np.uint8)

        elif method == 'adaptive' or method == 1.0:
            # 自适应阈值化，根据局部区域的灰度值计算阈值
            block_size = threshold_parameters_dict.get('block_size', 15)  # 默认块大小为11
            c = threshold_parameters_dict.get('c', 3)  # 默认常数值为2

            # 扩展为适用于自适应阈值化的操作
            padded_image = np.pad(image, block_size // 2, mode='constant', constant_values=0)
            binary_image = np.zeros_like(image)

            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    # 获取当前区域的块
                    block = padded_image[i:i + block_size, j:j + block_size]
                    local_threshold = np.mean(block) - c
                    binary_image[i, j] = 255 if image[i, j] > local_threshold else 0

            binary_image = binary_image.astype(np.uint8)

        elif method == 'otsu' or method == 2.0:
            # Otsu 自动阈值化
            # 计算图像的直方图
            hist, bin_edges = np.histogram(image, bins=256, range=(0, 256))
            total_pixels = image.size
            # 计算前景和背景的类间方差
            current_max, threshold_value = 0, 0
            sum_total = np.sum(np.arange(256) * hist)  # 总灰度值的加权和
            sum_bg, weight_bg, weight_fg = 0, 0, 0

            for t in range(256):
                weight_bg += hist[t]
                if weight_bg == 0:
                    continue
                weight_fg = total_pixels - weight_bg
                if weight_fg == 0:
                    break

                sum_bg += t * hist[t]
                mean_bg = sum_bg / weight_bg
                mean_fg = (sum_total - sum_bg) / weight_fg
                between_class_variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

                if between_class_variance > current_max:
                    current_max = between_class_variance
                    threshold_value = t

            # 使用 Otsu 算法得到的阈值进行二值化
            binary_image = np.where(image > threshold_value, 255, 0).astype(np.uint8)

        else:
            raise ValueError(f"Unsupported thresholding method: {method}")  # 不支持的阈值化方法
        
        return binary_image  # 返回二值化图像

    def convolution(self, convolution_parameters_dict:dict)    -> np.ndarray:
        """
        应用卷积操作并进行阈值化处理
        :param convolution_parameters_dict: 参数字典，结构为：{'kernel'：np.array,'threshold':threshold}
        """
        image = ImageProcessor(self.image.copy()).to_grayscale()  # 转换为灰度图像
        kernel, threshold = convolution_parameters_dict['kernel'], convolution_parameters_dict['threshold']  # 提取卷积核和阈值
        # 使用卷积核对图像进行卷积操作
        convolved_image = conv2d(image, kernel)
        # 对卷积结果进行阈值化处理
        binary_image = ImageProcessor(convolved_image).threshold_image({'threshold': threshold})
        return binary_image  # 返回经过卷积和阈值化处理后的二值图像

    def negative_blend(self, overlay_image: np.ndarray)    -> np.ndarray:
        """
        将当前图像和叠加图像进行负片相乘混合
        :param overlay_image: 被负片叠加的图像
        :return: 返回叠加后的图像
        """
        
        if overlay_image is None:
            raise ValueError("Overlay image is required.")
        
        image_type = ImageProcessor(overlay_image).type  # 获取叠加图像的类型

        if overlay_image.shape != self.image.shape:
            # 如果叠加图像和当前图像的尺寸不一致，调整尺寸
            overlay_image = resize_to(overlay_image, self.image.shape)

        if image_type != self.type:
            # 如果叠加图像的类型与当前图像不匹配，将其转换为相同类型
            overlay_image = overlay_image.astype(self.image.dtype)

        result = overlay_image.astype(np.int32) - np.array(self.image).astype(np.int32)  # 逐元素相减
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result  # 返回叠加后的图像

    def diffcheck(self, overlay_image: np.ndarray, threshold: int = 17) -> np.ndarray:
        """
        使用阈值提取原始图像和叠加图像之间的差异，并生成混合图像
        :param overlay_image: 被负片叠加的图像
        :param threshold: 像素差异的阈值，低于该值的差异不会被识别
        :return: 返回叠加后的图像
        """
        mask = subtract_images_with_abs(self.image, overlay_image)  # 计算图像之间的绝对差异
        mask = ImageProcessor(mask).threshold_image(threshold_parameters_dict={'threshold': threshold})  # 对差异图像进行阈值处理

        # 使用掩码展示图像差异部分
        result_image = show_mask(self.image, mask)

        return result_image  # 返回应用掩码后的结果图像

    def find_region(self, index: int, axis: int, condition) -> tuple[int, int]:
        """
        在图像中寻找指定行或列的目标区域的两个端点，支持自定义条件。

        :param axis: 0 或 1，表示是要查找的行还是列
        :param index: 指定的行号或列号
        :param condition: 自定义条件函数，该函数接收一个像素值，返回布尔值，默认为None表示查找灰度值大于173的区域

        :return: 匹配区域的两个端点 (start, end)，如果没有找到区域，返回 (None, None)
        """

        image = self.to_grayscale()
        if axis == 0:
            line = image[index, :]  # 获取指定行
        elif axis == 1:
            line = image[:, index]  # 获取指定列
        else:
            raise ValueError("axis should be 0 or 1")

        # 如果没有提供条件，默认使用较白色区域（像素值> 173）
        if condition is None:
            condition = lambda pixel: pixel > 173

        # 应用条件，找到符合条件的所有像素位置
        matching_pixels = np.where([condition(pixel) for pixel in line])[0]

        if len(matching_pixels) == 0:
            return -1, -1  # 没有找到目标区域

        # 找到匹配区域的起始和结束位置
        start = matching_pixels[0]
        end = matching_pixels[-1]

        return start, end

    def circle_recongnition(self, circle_parameters_dict:dict)  -> np.ndarray:
        for key, value in circle_parameters_dict.items():
            circle_parameters_dict[key] = int(value)
        grayscale_image = self.to_grayscale()

        circles = cv2.HoughCircles(
            grayscale_image,
            cv2.HOUGH_GRADIENT,
            dp=1,
            param1=circle_parameters_dict['param1'],
            param2=circle_parameters_dict['param2'],
            minDist=circle_parameters_dict['minDist'],
            minRadius=circle_parameters_dict['minRadius'],
            maxRadius=circle_parameters_dict['maxRadius'],
        )
        self.features_manager.remove_testing_features()
        if circles is not None:
            rgb_image = self.to_rgb()
            for circle in circles[0]:
                self.features_manager.add_feature("Circle", {"center": (int(circle[0]), int(circle[1])), 'radius': int(circle[2])})  # 存储识别到的圆
                cv2.circle(rgb_image, (int(circle[0]), int(circle[1])), int(circle[2]), (0, 255, 0), thickness=1)
            return rgb_image
        else:
            return self.image

    def line_recongition(self, line_parameters_dict:dict)  -> np.ndarray:
        for key, value in line_parameters_dict.items():
            line_parameters_dict[key] = int(value)
        grayscale_image = self.to_grayscale()
        lines = cv2.HoughLinesP(
            grayscale_image,
            1,
            np.pi / 360,
            threshold=line_parameters_dict['threshold'],
            minLineLength=line_parameters_dict['minLineLength'],
            maxLineGap=line_parameters_dict['maxLineGap']
        )
        self.features_manager.remove_testing_features()
        if lines is not None:
            rgb_image = self.to_rgb()
            for line in lines:
                line = np.array(line)[0]
                self.features_manager.add_feature("Line", {'start': (int(line[0]), int(line[1])), 'end': (int(line[2]), int(line[3]))})  # 存储识别到的直线
                cv2.line(rgb_image, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), (0, 255, 0),
                         thickness=1)
            return rgb_image
        else:
            return self.image

    def contour_recongition(self,contour_parameters_dict:dict)  -> np.ndarray:
        grayscale_image = self.to_grayscale()
        contours, hierarchy = cv2.findContours(grayscale_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.features_manager.remove_testing_features()
        if contours is not None:
            rgb_image = self.to_rgb()
            cv2.drawContours(rgb_image,contours, -1, (0, 0, 255), 0)
            for contour in contours:

                n=len(contour)
                print(contour, n)
                self.features_manager.add_feature("Contour", {"contour": np.array(contour).reshape((n, 2)).tolist()})
            return rgb_image
        else:
            return self.image


class Feature:
    def __init__(self, feature_type: str = "None", data: dict = {}, status: str = "testing"):
        self.type = feature_type
        self.data = data
        self.status = status

    def __str__(self):
        return f"Type: {self.type}, Data: {self.data}, Status: {self.status}"

    def feature_distance(self, coordinates : tuple) -> float:
        if self.type == "Circle":
            center_x, center_y = self.data['center']
            radius = self.data['radius']
            return np.sqrt((center_x - coordinates[0]) ** 2 + (center_y - coordinates[1]) ** 2) - radius
        elif self.type == "Line":
            start_x, start_y = self.data['start']
            end_x, end_y = self.data['end']
            return np.abs((end_y - start_y) * coordinates[0] - (end_x - start_x) * coordinates[1] +
                       end_x * start_y - end_y * start_x) / np.sqrt((end_y - start_y) ** 2 + (end_x - start_x) ** 2)
        elif self.type == "Contour":
            min_distance = np.inf
            for point in self.data['contour']:
                distance = np.sqrt((point[0]-coordinates[0])**2+(point[1]-coordinates[1])**2)
                if distance < min_distance:
                    min_distance = distance
            return min_distance
        else:
            raise ValueError(f"Unsupported feature type: {self.type}")
        # 添加其他特征类型的处理逻辑

    def line_contour_intersections(self, line_segment: tuple[tuple,tuple], contour: list) -> list:
        """
        Find intersection points between a line segment and a contour.

        Parameters:
        - line_segment: tuple, containing two points (x1, y1), (x2, y2) representing the line segment
        - contour: numpy.ndarray, contour points

        Returns:
        - list of tuples, each tuple is an intersection point (x, y)
        """

        def line_intersection(line1, line2):
            xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
            ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

            def det(a, b):
                return a[0] * b[1] - a[1] * b[0]

            div = det(xdiff, ydiff)
            if div == 0:
                return None  # Lines don't intersect

            d = (det(*line1), det(*line2))
            x = det(d, xdiff) / div
            y = det(d, ydiff) / div
            return x, y

        intersections = []
        for i in range(len(contour)):
            pt1 = contour[i][0]
            pt2 = contour[(i + 1) % len(contour)][0]  # Loop back to the first point
            intersection = line_intersection(line_segment, ((pt1[0], pt1[1]), (pt2[0], pt2[1])))
            if intersection:
                if is_point_on_line(intersection, line_segment) and is_point_on_line(intersection, (pt1, pt2)):
                    intersections.append(intersection)
        return intersections


class FeatureManager:
    def __init__(self, features: list[Feature] = []):
        self.features = features

    def __str__(self):
        feature_strings = [str(feature) for feature in self.features]
        return f"Features: {feature_strings}"

    def add_feature(self, feature_type: str, data: dict) -> 'FeatureManager':
        feature = Feature(feature_type, data)
        self.features.append(feature)
        return self

    def remove_testing_features(self) -> 'FeatureManager':
        self.features = [feature for feature in self.features if feature.status != "testing"]
        return self

    def confirmed(self):
        for feature in self.features:
            if feature.status == "testing":
                feature.status = "confirmed"
        return self

    def get_nearest_feature(self, coordinates: tuple) -> Feature:
        nearest_feature = Feature()
        min_distance = np.inf

        for feature in self.features:
            distance = feature.feature_distance(coordinates)
            if distance < min_distance:
                min_distance = distance
                nearest_feature = feature

        return nearest_feature

    def save_to_csv(self, save_path: str):
        fieldnames = ['type', 'data', 'status']
        with open(save_path, 'w', newline='', encoding='utf-8') as file:
            # 写入标题行
            file.write(';'.join(fieldnames) + '\n')

            # 写入数据到CSV文件
            for feature in self.features:
                row = [feature.type, str(feature.data), feature.status]
                file.write(';'.join(row) + '\n')
    
    def load_from_csv(self, load_path: str):
        self.features = []
        with open(load_path, 'r', newline='', encoding='utf-8') as file:
            lines = file.readlines()
            print(lines)
            for line in lines[1:]:
                feature_type, data, status = line.strip().split(';')
                feature = Feature(feature_type, eval(data), status)
                self.features.append(feature)


def subtract_images_with_abs(overlay_image:np.ndarray, target_image:np.ndarray) -> np.ndarray:
    """
    计算两幅图像的绝对差异，并返回结果图像。

    该函数接受两幅具有相同尺寸的图像，计算目标图像与叠加图像的像素值差异，并返回每个像素的绝对差异。
    结果图像的像素值范围被限制在 [0, 255] 之间。

    参数:
    overlay_image (np.ndarray): 第一幅输入图像（叠加图像），类型为 NumPy 数组。
    target_image (np.ndarray): 第二幅输入图像（目标图像），类型为 NumPy 数组。

    返回:
    np.ndarray: 包含输入图像之间绝对差异的新图像（NumPy 数组）。

    异常:
    ValueError: 如果输入图像的尺寸不相同，则抛出异常。
    """
    # 确保图像尺寸相同
    if overlay_image.shape != target_image.shape:
        raise ValueError("图像尺寸必须相同。")

    # 创建一个与输入图像具有相同形状和数据类型的空结果图像
    result_image = np.zeros_like(overlay_image)

    # 遍历每个像素
    for y in range(overlay_image.shape[0]):
        for x in range(overlay_image.shape[1]):
            for c in range(overlay_image.shape[2]):  # 遍历每个通道（RGB）
                # 计算差异并取绝对值
                diff = np.abs(int(overlay_image[y, x, c]) - int(target_image[y, x, c]))

                # 确保结果在 [0, 255] 范围内
                result_image[y, x, c] = min(255, diff)

    return result_image


def show_mask(image:np.ndarray, mask:np.ndarray) -> np.ndarray:
    """
    将掩码应用于图像，将掩码为零的像素设置为零。

    参数:
    image (np.ndarray): 输入图像，形状为 (height, width, channels) 的 NumPy 数组。
    mask (np.ndarray): 掩码，形状为 (height, width) 的 NumPy 数组。非零值表示在结果图像中保留的像素。

    返回:
    np.ndarray: 应用了掩码的结果图像。掩码为零的像素在结果图像中也被设置为零。
    """
    result_image = np.zeros_like(image)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):  # Iterate over each channel (RGB)
                # Perform subtraction and take the absolute value
                if int(mask[y, x]):
                    result_image[y, x, c] = image[y, x, c]
                else:
                    result_image[y, x, c] = 0
    return result_image


def is_point_on_line(point:tuple, line:tuple[tuple,tuple]) -> bool:
    """
    检查一个点是否在一条线段上。

    :参数:
    - point: tuple, 点 (x, y)
    - line: tuple, 包含两个点 (x1, y1), (x2, y2) 表示线段

    返回:
    - bool, 如果点在线段上返回 True，否则返回 False
    """
    x, y = point
    (x1, y1), (x2, y2) = line
    cross_product = (y - y1) * (x2 - x1) - (x - x1) * (y2 - y1)
    if np.abs(cross_product) > 1e-6:
        return False  # (x, y) is not on the line

    dot_product = (x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)
    if dot_product < 0:
        return False  # (x, y) is on the line but not between (x1, y1) and (x2, y2)

    squared_length = (x2 - x1) ** 2 + (y2 - y1) ** 2
    if dot_product > squared_length:
        return False  # (x, y) is on the line but not between (x1, y1) and (x2, y2)

    return True


def resize_to(image:np.ndarray, target_shape:tuple) -> np.ndarray:
    """
    调整图像大小或填充图像以匹配目标形状。

    参数:
    image (np.ndarray): 输入图像。
    target_shape (tuple): 目标形状 (高度, 宽度)。

    返回:
    np.ndarray: 调整大小或填充后的图像，具有目标形状。

    注意:
    - 如果输入图像小于目标形状，将用零填充。
    - 如果输入图像大于目标形状，将裁剪以适应。
    - 如果输入图像已经匹配目标形状，将按原样返回。
    """
    height, width = target_shape[:2]

    if image.shape[:2] == target_shape[:2]:
        # If the shapes are already equal, return the original image
        return image

    # Resize or pad the image to match the target shape
    resized_or_padded_image = np.zeros(target_shape, dtype=np.uint8)

    if image.shape[0] < height or image.shape[1] < width:
        # If image is smaller than target shape, pad the image
        resized_or_padded_image[:image.shape[0], :image.shape[1]] = image
    else:
        # If image is larger than target shape, crop the image
        resized_or_padded_image = image[:height, :width]

    return resized_or_padded_image


def conv2d(image:np.ndarray, kernel:np.ndarray) -> np.ndarray:
    """对图像应用二维卷积"""
    # 获取图像和卷积核的尺寸
    kernel = np.array(kernel)
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # 计算卷积操作时的填充大小（为了保持图像尺寸不变，使用半个卷积核大小的填充）
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    pad_value = ((pad_height, pad_height), (pad_width, pad_width))
    # 对图像进行零填充
    padded_image = np.pad(image, pad_value, mode='constant', constant_values=0)

    # 准备一个空的输出图像
    output_image = np.zeros_like(image)

    # 遍历每个像素，执行卷积操作
    for i in range(image_height):
        for j in range(image_width):
            # 获取当前位置的邻域区域
            region = padded_image[i:i + kernel_height, j:j + kernel_width]
            # 卷积操作：将邻域区域与卷积核进行逐元素相乘并求和
            output_image[i, j] = np.sum(region * kernel)

    return output_image

def imread(filename: str) -> np.ndarray:
    with Image.open(filename) as img:
        img_array = np.array(img.convert('RGB'))
    return img_array

def imwrite(filename: str, img_array: np.ndarray):
    # Convert numpy array to PIL Image
    img = Image.fromarray(img_array)
    # Save the image
    img.save(filename)
