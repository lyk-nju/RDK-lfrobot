import cv2 as cv
import uuid
import os
import numpy as np

class ImageProcessor:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.x = -1
        self.y = -1
        self.uuid = -1
        self.image = np.zeros((640, 224, 3))
        self.current_image = None
        self.image_files = []
        self.current_index = 0

        # 检查输入输出目录
        if not os.path.exists(self.input_dir):
            raise FileNotFoundError(f"输入目录不存在: {self.input_dir}")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # 获取所有图像文件
        self.image_files = [f for f in os.listdir(self.input_dir) 
                          if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        self.image_files.sort()
        if not self.image_files:
            raise FileNotFoundError(f"在目录 {self.input_dir} 中未找到图像文件")

        # 初始化窗口
        cv.namedWindow("capture image", cv.WINDOW_NORMAL)
        cv.resizeWindow("capture image", 640, 224)

        # 加载第一张图像
        self.load_next_image()

    def mouse_callback(self, event, x, y, flags, userdata):
        if event == cv.EVENT_LBUTTONDOWN:
            imageWithCircle = self.current_image.copy()
            self.x = x
            self.y = y
            cv.circle(imageWithCircle, (x, y), 5, (0, 0, 255), -1)
            cv.imshow("capture image", imageWithCircle)

    def load_next_image(self):
        if self.current_index >= len(self.image_files):
            print("已处理所有图像")
            return False

        file_path = os.path.join(self.input_dir, self.image_files[self.current_index])
        print(f"正在处理: {file_path}")
        
        # 读取并裁剪图像（假设原始图像为640x480）
        full_image = cv.imread(file_path)
        if full_image is None:
            print(f"无法读取图像: {file_path}")
            return False
            
        # 裁剪ROI区域 (128:352, 即垂直方向中间224像素)
        self.current_image = full_image[128:352, :, :].copy()
        cv.setMouseCallback("capture image", self.mouse_callback, self.current_image)
        cv.imshow("capture image", self.current_image)
        return True

    def process_images(self):
        while self.current_index < len(self.image_files):
            key = cv.waitKey(0)
            
            # 回车键保存并继续
            if key == 32 or key == 13:  # 空格键 or 回车
                if self.x != -1 and self.y != -1:
                    self.uuid = f'xy_{self.x:03d}_{self.y:03d}_{uuid.uuid1()}'
                    output_path = os.path.join(self.output_dir, self.uuid + '.jpg')
                    cv.imwrite(output_path, self.current_image)
                    print(f"已保存: {output_path}")

                # 加载下一张图像
                self.current_index += 1
                self.x = -1
                self.y = -1
                if not self.load_next_image():
                    break
                    
            # ESC键退出
            elif key == 27:  
                break

        cv.destroyAllWindows()

def main(args=None):
    # 配置参数
    input_directory = "./input_images"  # 原始图像目录
    output_directory = "./image_dataset"  # 输出目录

    try:
        processor = ImageProcessor(input_directory, output_directory)
        processor.process_images()
        print("处理完成")
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == '__main__':
  main()