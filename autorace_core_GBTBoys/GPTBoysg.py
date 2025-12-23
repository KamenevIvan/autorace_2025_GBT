import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from std_msgs.msg import String
import numpy as np
import cv2
import math

class Competition(Node):
    def __init__(self):
        super().__init__('competition')
        
        self.declare_parameter('tolerance_percent', 0.075)
        self.declare_parameter('linear_speed', 0.14)
        self.declare_parameter('track_color', [40, 40, 40])
        self.tolerance_percent = self.get_parameter('tolerance_percent').value
        self.linear_speed = self.get_parameter('linear_speed').value
        self.track_color = np.array(self.get_parameter('track_color').value, dtype=np.uint8)
        
        self.phaze = 0
        self.direction = 0
        
        self.subscription = self.create_subscription(Image, '/color/image', self.image_callback, 10)
        
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.bridge = CvBridge()
        
    def turning_direction(self, cv_image):
        """Обнаружение синего знака и определение направления стрелки"""
        height, width, _ = cv_image.shape
        
        center_start_x = int(width * 0.25)
        center_end_x = int(width * 0.75)
        center_start_y = int(height * 0.10)
        center_end_y = int(height * 0.75)
        
        center_region = cv_image[center_start_y:center_end_y, center_start_x:center_end_x]
        
        hsv = cv2.cvtColor(center_region, cv2.COLOR_BGR2HSV)
        
        lower_blue1 = np.array([90, 50, 50])
        upper_blue1 = np.array([130, 255, 255])
        lower_blue2 = np.array([130, 50, 50])
        upper_blue2 = np.array([150, 255, 255])
        
        blue_mask1 = cv2.inRange(hsv, lower_blue1, upper_blue1)
        blue_mask2 = cv2.inRange(hsv, lower_blue2, upper_blue2)
        blue_mask = cv2.bitwise_or(blue_mask1, blue_mask2)
        
        kernel = np.ones((5, 5), np.uint8)
        blue_mask = cv2.erode(blue_mask, kernel, iterations=1)
        blue_mask = cv2.dilate(blue_mask, kernel, iterations=2)
        
        contours_result = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours_result) == 3:
            contours = contours_result[1]
        else:
            contours = contours_result[0]
        
        if not contours:
            return False, None, 0, None
        
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area > 200:
            x, y, w, h = cv2.boundingRect(largest_contour)
            full_x = x + center_start_x
            full_y = y + center_start_y
            
            perimeter = cv2.arcLength(largest_contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0
            
            is_circular = circularity > 0.7
            arrow_direction = None
            
            if is_circular:
                # Создаем маску круга для изоляции содержимого знака
                circle_mask = np.zeros_like(blue_mask)
                cv2.drawContours(circle_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
                
                # Применяем маску круга к HSV центральной области
                masked_center_hsv = cv2.bitwise_and(hsv, hsv, mask=circle_mask)
                
                # Вырезаем подобласть для bounding rect (для совместимости с дальнейшей логикой)
                masked_sign_hsv = masked_center_hsv[y:y+h, x:x+w]
                
                # Для дебаг: sign_region остается как bounding rect оригинального изображения
                sign_region = cv_image[full_y:full_y+h, full_x:full_x+w]
                
                if sign_region.size > 0:
                    white_l = np.array([0, 0, 70])  
                    white_u = np.array([180, 40, 255])  
                    
                    white_mask = cv2.inRange(masked_sign_hsv, white_l, white_u)
                    
                    # Если маска пустая, fallback на adaptive threshold
                    if cv2.countNonZero(white_mask) < 100:  # Если мало пикселей — adaptive
                        gray_sign = cv2.cvtColor(sign_region, cv2.COLOR_BGR2GRAY)
                        adaptive_mask = cv2.adaptiveThreshold(gray_sign, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
                        
                        # Вырезаем sub_circle_mask для области bounding rect
                        sub_circle_mask = circle_mask[y:y+h, x:x+w]
                        
                        white_mask = cv2.bitwise_and(adaptive_mask, adaptive_mask, mask=sub_circle_mask)
                    
                    # Уменьшаем erode для стрелки
                    if w < 90:
                        erode_iter = 0
                        dilate_iter = 1
                        kernel_white = np.ones((3, 3), np.uint8)
                    else:
                        erode_iter = 0  # Убрали erode, чтобы не стирать стрелку
                        dilate_iter = 2
                        kernel_white = np.ones((5, 5), np.uint8)
                    
                    white_mask = cv2.erode(white_mask, kernel_white, iterations=erode_iter)
                    white_mask = cv2.dilate(white_mask, kernel_white, iterations=dilate_iter)
                    
                    # Вывод только решающей маски
                    # cv2.imshow('Final Arrow Mask (Isolated Circle)', white_mask)
                    # cv2.waitKey(1)
                    
                    lines = cv2.HoughLinesP(white_mask, rho=1, theta=np.pi/180, threshold=20, minLineLength=10, maxLineGap=5)
                    if lines is not None:
                        angles = []
                        for line in lines:
                            x1, y1, x2, y2 = line[0]
                            ang = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                            angles.append(ang)
                        avg_angle = np.mean(angles)
                        if avg_angle > 0:
                            arrow_direction = "LEFT"
                        else:
                            arrow_direction = "RIGHT"
                        
                    else:
                        arrow_direction = None
                    
                    white_contours_result = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if len(white_contours_result) == 3:
                        white_contours = white_contours_result[1]
                    else:
                        white_contours = white_contours_result[0]
                    
                    if white_contours:
                        arrow_contour = max(white_contours, key=cv2.contourArea)
                        arrow_x, arrow_y, arrow_w, arrow_h = cv2.boundingRect(arrow_contour)
                        
                        sign_center_x = w // 2
                        sign_center_y = h // 2
                        arrow_center_x = arrow_x + arrow_w // 2

                        rect = cv2.minAreaRect(arrow_contour)
                        angle = rect[2]
                        if angle > -45:
                            arrow_direction = "LEFT"
                        else:
                            arrow_direction = "RIGHT"

                        # Начальная по bounding
                        if arrow_center_x < sign_center_x:
                            arrow_direction = "LEFT"
                        else:
                            arrow_direction = "RIGHT"
                        
                        # Центр массы
                        M = cv2.moments(arrow_contour)
                        if M['m00'] > 0:
                            cx_arrow = int(M['m10'] / M['m00'])
                            cy_arrow = int(M['m01'] / M['m00'])
                            if cx_arrow < sign_center_x:
                                arrow_direction = "LEFT"  # Убрали swap
                            else:
                                arrow_direction = "RIGHT"
                            
                            # Дополнительная: пиксели слева/справа (overrides)
                            left_half = white_mask[:, :sign_center_x]
                            right_half = white_mask[:, sign_center_x:]
                            left_pixels = cv2.countNonZero(left_half)
                            right_pixels = cv2.countNonZero(right_half)
                            if left_pixels + right_pixels < 200:
                                arrow_direction = None
                            elif left_pixels > right_pixels:
                                arrow_direction = "RIGHT"  # Убрали swap и инвертировали
                            else:
                                arrow_direction = "LEFT"
                        
                        # Основание + две красные линии + пересечения (override)
                        row_sums = np.sum(white_mask, axis=1)
                        if np.any(row_sums > 0):
                            bottom_y = np.where(row_sums > 0)[0][-1]  # Самая нижняя строка с белыми пикселями
                            bottom_row = white_mask[bottom_y, :]
                            if np.any(bottom_row > 0):
                                min_x = np.min(np.where(bottom_row > 0)[0])
                                max_x = np.max(np.where(bottom_row > 0)[0])
                                center_x = (min_x + max_x) // 2
                                
                                height_up = h // 5  # Изменение: адаптируем под размер (было min(20, h//4), теперь больше для лучших sums)
                                offset = w // 5  # Изменение: уменьшаем offset для лучшего захвата на малых
                                
                                left_line_x = max(0, center_x - offset)
                                right_line_x = min(w - 1, center_x + offset)
                                
                                left_sum = 0
                                for dy in range(1, height_up + 1):
                                    y = bottom_y - dy
                                    if y >= 0:
                                        if white_mask[y, left_line_x] > 0:
                                            left_sum += 1
                                
                                right_sum = 0
                                for dy in range(1, height_up + 1):
                                    y = bottom_y - dy
                                    if y >= 0:
                                        if white_mask[y, right_line_x] > 0:
                                            right_sum += 1
                                
                                if left_sum > right_sum:
                                    arrow_direction = "RIGHT"  # Инвертировали (для вправо)
                                elif right_sum > left_sum:
                                    arrow_direction = "LEFT"  # Инвертировали (для влево)
                                else:
                                    arrow_direction = None  # Можно добавить fallback: if None, использовать Hough (arrow_direction remains from previous)

            debug_img = cv_image.copy()
            cv2.rectangle(debug_img, (center_start_x, center_start_y), 
                        (center_end_x, center_end_y), (255, 0, 0), 2)
            
            if area > 4000:
                color = (0, 255, 0) if is_circular else (0, 255, 255)
                cv2.rectangle(debug_img, (full_x, full_y), (full_x + w, full_y + h), color, 2)
                
                info_text = f'Blue: {area:.0f}px, Circle: {circularity:.2f}'
                if arrow_direction:
                    info_text += f', Arrow: {arrow_direction}'
                
                cv2.putText(debug_img, info_text, (full_x, full_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                if is_circular:
                    cv2.rectangle(debug_img, (full_x, full_y), (full_x + w, full_y + h), (0, 255, 0), 3)
                    cv2.putText(debug_img, 'CIRCULAR SIGN!', (full_x, full_y - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    if arrow_direction:
                        direction_text = f'DIRECTION: {arrow_direction}'
                        text_color = (0, 165, 255) if arrow_direction == "LEFT" else (0, 255, 165)
                        cv2.putText(debug_img, direction_text, (full_x, full_y + h + 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
            
            #cv2.imshow('Blue Sign Detection', debug_img)
            #cv2.waitKey(1)
            
            return True, (full_x, full_y, w, h), area, arrow_direction
        
        return False, None, 0, None
    
    def surrounding_color_checker(self, image, color, threshold, tolerance_percent):
        """Данная функция предназначена для поиска и сопоставления пикселей"""
        max_diff = 255 * tolerance_percent
        diff = np.abs(image.astype(np.float32) - color.astype(np.float32))
        matches = np.all(diff <= max_diff, axis=2)
        percentage = np.sum(matches) / (image.shape[0] * image.shape[1])
        return percentage >= threshold, percentage

    def get_point_color(self, image, point, radius=5):
        """Данная функция вычисляет цвет изображения в некотором радиусе от точки"""
        x, y = int(point[0]), int(point[1])
        
        x_min = max(0, x - radius)
        x_max = min(image.shape[1], x + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(image.shape[0], y + radius + 1)
        
        region = image[y_min:y_max, x_min:x_max]
        
        if region.size > 0:
            avg_color = np.mean(region, axis=(0, 1))
        else:
            avg_color = np.array([0, 0, 0], dtype=np.float32)
            
        return avg_color.astype(np.uint8)
        
    def close_clolor_checker(self, color1, color2, tolerance=10):
        """Схожесть цветов"""
        diff = np.abs(color1.astype(np.int16) - color2.astype(np.int16))
        return np.all(diff <= tolerance)

    def road_sabilizer(self, image, base_speed, base_angular=0.0):
        """Стабилизация робота на дороге через трапецию"""
        height, width = image.shape[:2]
        points_cords = {
            '1': (250, height - 5), 
            '2': (width - 250, height - 5), 
            '3': (200, height - 20), 
            '4': (width - 200, height - 20) 
        }

        points_colors = {
            '1': self.get_point_color(image, points_cords['1'], radius=5),
            '2': self.get_point_color(image, points_cords['2'], radius=5),
            '3': self.get_point_color(image, points_cords['3'], radius=5),
            '4': self.get_point_color(image, points_cords['4'], radius=5)
        }

        angle_to_rotate = 0.0
        if not self.close_clolor_checker(points_colors['1'], self.track_color):
            angle_to_rotate -= 2.5
        if not self.close_clolor_checker(points_colors['2'], self.track_color):
            angle_to_rotate += 2.5
        if not self.close_clolor_checker(points_colors['3'], self.track_color):
            angle_to_rotate -= 5.0
        if not self.close_clolor_checker(points_colors['4'], self.track_color):
            angle_to_rotate += 5.0
        
        cmd = Twist()
        cmd.linear.x = base_speed
        cmd.angular.z = math.radians(angle_to_rotate)
        
        cmd.angular.z += math.radians(base_angular)
        if not cmd.angular.z == 0.0:
            cmd.linear.x = base_speed / 4
        
        if angle_to_rotate != 0.0:
            cmd.linear.x = 0.01
        return cmd

    def turning_dir_helper(self, image):
        """Вспомогательная фунция определения знака
        Работает через определение синего цвета"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=100,
            param1=50,
            param2=30,
            minRadius=30,
            maxRadius=min(h, w) // 2
        )
        
        if circles is None:
            return 0
         
        circle = circles[0][0]
        x, y, r = int(circle[0]), int(circle[1]), int(circle[2])

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (x, y), r-10, 255, -1)
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        
        x1 = max(0, x - r)
        y1 = max(0, y - r)
        x2 = min(w, x + r)
        y2 = min(h, y + r)
        
        cropped = masked_image[y1:y2, x1:x2]
        if cropped.size == 0:
            return 0

        lower_white = np.array([100, 100, 0])
        upper_white = np.array([255, 255, 255])
        binary = cv2.inRange(cropped, lower_white, upper_white)
        
        binary_with_stripe = binary.copy()
        binary_with_stripe[:r+10, :] = 0

        left_part = binary_with_stripe[:, :r] 
        right_part = binary_with_stripe[:, -r:] 
        
        white_pixels_left = np.sum(left_part == 255)
        white_pixels_right = np.sum(right_part == 255)
        
        if white_pixels_left < white_pixels_right:
            return 1  
        else:
            return -1   

    def image_callback(self, msg): 
        """Основная функция управления"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            
            if self.phaze == 0:
                result, percentage = self.surrounding_color_checker(
                    cv_image, 
                    np.array([0, 230, 0], dtype=np.uint8),
                    0.01,
                    self.tolerance_percent
                )
                if result:
                    self.phaze = 1
                else:
                    return
            
            if self.phaze == 1:
                result, percentage = self.surrounding_color_checker(
                    cv_image, 
                    np.array([0, 55, 90], dtype=np.uint8),
                    0.0175,
                    self.tolerance_percent
                )
                if result:
                    cmd = Twist()
                    cmd.linear.x = 0.0
                    self.publisher.publish(cmd)
                    self.phaze = 2
                else:
                    cmd = self.road_sabilizer(cv_image, self.linear_speed)
                    self.publisher.publish(cmd)
                    return
            if self.phaze == 2:
                cv_crop = cv_image[:, 350:-350]
                result, percentage = self.surrounding_color_checker(
                    cv_crop, 
                    np.array([0, 55, 90], dtype=np.uint8),
                    0.05,
                    self.tolerance_percent
                )
                if result:
                    cmd = Twist()
                    cmd.linear.x = 0.0
                    self.publisher.publish(cmd)
                    self.phaze = 3
                else:
                    cmd = Twist()
                    cmd.angular.z = math.radians(2.5)
                    cmd.linear.x = 0.0
                    self.publisher.publish(cmd)
                    return
            if self.phaze == 3:
                self.direction = self.turning_dir_helper(cv_image)
                if self.direction == -1:
                    a = 1
                    a += 1
                elif self.direction == 1:
                    a = 1
                    a += 1
                else:
                    self.direction == 1
                self.phaze = 4
            if self.phaze == 4:
                result, percentage = self.surrounding_color_checker(
                    cv_image, 
                    np.array([255, 100, 50], dtype=np.uint8),
                    0.004,
                    self.tolerance_percent
                )
                if result:
                    cmd = Twist()
                    cmd.linear.x = 0.0
                    self.publisher.publish(cmd)
                    self.phaze = 5
                    
                    pub = self.create_publisher(String, 'robot_finish', 10)
                    msg = String()
                    msg.data = "GBTBoys"
                    pub.publish(msg)
                else:
                    cmd = self.road_sabilizer(cv_image, self.linear_speed, self.direction * 2.5)
                    self.publisher.publish(cmd)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = Competition()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        
        stop_msg = Twist()
        node.publisher.publish(stop_msg)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()