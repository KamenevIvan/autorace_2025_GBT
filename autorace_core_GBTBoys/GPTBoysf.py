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


        self.phase = 0
        self.direction = 0
        self.declare_parameter('tolerance_percent', 0.05)
        self.declare_parameter('linear_speed', 0.02)
        self.declare_parameter('track_color', [40, 40, 40])
        
        self.tolerance_percent = self.get_parameter('tolerance_percent').value
        self.linear_speed = self.get_parameter('linear_speed').value
        self.track_color = np.array(self.get_parameter('track_color').value, dtype=np.uint8)
        
        self.subscription = self.create_subscription(Image, '/color/image', self.image_callback, 10)
        
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.bridge = CvBridge()
        
    
    
    def check_color_percentage(self, image, color, threshold, tolerance_percent):
        max_diff = 255 * tolerance_percent
        diff = np.abs(image.astype(np.float32) - color.astype(np.float32))
        matches = np.all(diff <= max_diff, axis=2)
        percentage = np.sum(matches) / (image.shape[0] * image.shape[1])
        return percentage >= threshold, percentage

    def get_point_color(self, image, point, radius=5):
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
        
    def is_color_similar(self, color1, color2, tolerance=10):
        diff = np.abs(color1.astype(np.int16) - color2.astype(np.int16))
        return np.all(diff <= tolerance)

    def track_following(self, image, base_speed, base_angular=0.0):
        height, width = image.shape[:2]
        points_cords = {
            'BL': (250, height - 5), 
            'BR': (width - 250, height - 5),
            'TL': (200, height - 20), 
            'TR': (width - 200, height - 20) 
        }

        points_colors = {
            'BL': self.get_point_color(image, points_cords['BL'], radius=5),
            'BR': self.get_point_color(image, points_cords['BR'], radius=5),
            'TL': self.get_point_color(image, points_cords['TL'], radius=5),
            'TR': self.get_point_color(image, points_cords['TR'], radius=5)
        }

        angle_to_rotate = 0.0
        if not self.is_color_similar(points_colors['BL'], self.track_color):
            angle_to_rotate -= 2.5
        if not self.is_color_similar(points_colors['BR'], self.track_color):
            angle_to_rotate += 2.5
        if not self.is_color_similar(points_colors['TL'], self.track_color):
            angle_to_rotate -= 5.0
        if not self.is_color_similar(points_colors['TR'], self.track_color):
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

    def detect_turn_direction(self, image):
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
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            
            if self.phase == 0:
                result, percentage = self.check_color_percentage(
                    cv_image, 
                    np.array([0, 230, 0], dtype=np.uint8),
                    0.01,
                    self.tolerance_percent
                )
                if result:
                    self.phase = 1
                else:
                    return
            
            if self.phase == 1:
                result, percentage = self.check_color_percentage(
                    cv_image, 
                    np.array([0, 55, 90], dtype=np.uint8),
                    0.0175,
                    self.tolerance_percent
                )
                if result:
                    cmd = Twist()
                    cmd.linear.x = 0.0
                    self.publisher.publish(cmd)
                    self.phase = 2
                else:
                    cmd = self.track_following(cv_image, self.linear_speed)
                    self.publisher.publish(cmd)
                    return
            if self.phase == 2:
                cv_crop = cv_image[:, 350:-350]
                result, percentage = self.check_color_percentage(
                    cv_crop, 
                    np.array([0, 55, 90], dtype=np.uint8),
                    0.075,
                    self.tolerance_percent
                )
                if result:
                    cmd = Twist()
                    cmd.linear.x = 0.0
                    self.publisher.publish(cmd)
                    self.phase = 3
                else:
                    cmd = Twist()
                    cmd.angular.z = math.radians(2.5)
                    cmd.linear.x = 0.0
                    self.publisher.publish(cmd)
                    return
            if self.phase == 3:
                self.direction = self.detect_turn_direction(cv_image)
                if self.direction == -1:
                    asd = 1
                    asd +=1
                elif self.direction == 1:
                    asd = 1
                    asd -=1
                else:
                    self.direction == 1
                self.phase = 4
            if self.phase == 4:
                result, percentage = self.check_color_percentage(
                    cv_image, 
                    np.array([255, 100, 50], dtype=np.uint8),
                    0.002,
                    self.tolerance_percent
                )
                if result:
                    cmd = Twist()
                    cmd.linear.x = 0.0
                    self.publisher.publish(cmd)
                    
                    self.phase = 5
                else:
                    cmd = self.track_following(cv_image, self.linear_speed, self.direction * 2.5)
                    self.publisher.publish(cmd)
                    
                    pub = self.create_publisher(String, 'robot_finish', 10)
                    msg = String()
                    msg.data = "Gemini Boys"
                    pub.publish(msg)

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