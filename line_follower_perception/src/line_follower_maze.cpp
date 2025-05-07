//  该文件是line_follower_maze节点的源代码文件，其中包含了节点的所有功能的具体实现。
//  该节点的主要功能是接收图像数据，使用深度学习模型识别线条位置，并根据线条位置控制机器人的运动，实现自主巡线。
// Copyright (c) 2024, www.guyuehome.com
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "line_follower_perception/line_follower_perception.h" 

#include <fstream>
#include <string>   
#include <opencv2/opencv.hpp> 

#include "dnn_node/util/image_proc.h"
#include "hobot_cv/hobotcv_imgproc.h" 

#include <queue> 

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>  
#include "geometry_msgs/msg/quaternion.hpp" 
#include "nav_msgs/msg/odometry.hpp" 
#include "std_msgs/msg/bool.hpp"

typedef struct
{
  float roll;   // 翻滚角
  float pitch;  // 俯仰角
  float yaw;    // 偏航角
} DataImu;
enum // 定义枚举类型
{
  DIRECTION_LEFT = 0,     // 左转
  DIRECTION_RIGHT = 1,    // 右转
  DIRECTION_FRONT = 2,    // 前进
  DIRECTION_BACK = 3,     //转弯
};

/*外部暴露参数*/
float turning_vth_ = 1.0;               // 转弯速度
float straight_turning_vth_ = 0.08;     // 转弯速度
float straight_vth_ = 1.5;              // 直行速度
float go_time_ = 2.3;                   // 前进时间
std::string data_path_ = "/userdata/dev_front/parse_res.txt"; // 数据路径

/*内部控制参数*/
DataImu rpy[2];                          // 姿态数据
bool IsTurn, IsGo, stop_car_;            // 前进状态
float GoedTime;                          // 前进时间
std::queue<std::string> stringQueue;     // 动作队列
rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscription_; // 里程计订阅者
void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg); // 里程计回调函数


// 图片处理部分
void prepare_nv12_tensor_without_padding(const char *image_data, // 准备 NV12 格式的 tensor，不进行 padding
                                         int image_height, // 图像高度
                                         int image_width, // 图像宽度
                                         hbDNNTensor *tensor) // tensor
{
  auto &properties = tensor->properties; // 获取 tensor 属性
  properties.tensorType = HB_DNN_IMG_TYPE_NV12; // 设置 tensor 类型为 NV12 图像
  properties.tensorLayout = HB_DNN_LAYOUT_NCHW; // 设置 tensor 布局为 NCHW
  auto &valid_shape = properties.validShape; // 获取有效形状
  valid_shape.numDimensions = 4; // 设置维度数量为 4
  valid_shape.dimensionSize[0] = 1; // 设置 batch size 为 1
  valid_shape.dimensionSize[1] = 3; // 设置通道数为 3
  valid_shape.dimensionSize[2] = image_height; // 设置图像高度
  valid_shape.dimensionSize[3] = image_width; // 设置图像宽度

  auto &aligned_shape = properties.alignedShape; // 获取对齐后的形状
  aligned_shape = valid_shape; // 对齐后的形状与有效形状相同

  int32_t image_length = image_height * image_width * 3 / 2; // 计算图像数据长度

  hbSysAllocCachedMem(&tensor->sysMem[0], image_length); // 分配缓存内存
  memcpy(tensor->sysMem[0].virAddr, image_data, image_length); // 拷贝图像数据

  hbSysFlushMem(&(tensor->sysMem[0]), HB_SYS_MEM_CACHE_CLEAN); // 刷新缓存
}

void prepare_nv12_tensor_without_padding(int image_height, // 准备 NV12 格式的 tensor，不进行 padding
                                         int image_width, // 图像宽度
                                         hbDNNTensor *tensor) // tensor
{
  auto &properties = tensor->properties; // 获取 tensor 属性
  properties.tensorType = HB_DNN_IMG_TYPE_NV12; // 设置 tensor 类型为 NV12 图像
  properties.tensorLayout = HB_DNN_LAYOUT_NCHW; // 设置 tensor 布局为 NCHW

  auto &valid_shape = properties.validShape; // 获取有效形状
  valid_shape.numDimensions = 4; // 设置维度数量为 4
  valid_shape.dimensionSize[0] = 1; // 设置 batch size 为 1
  valid_shape.dimensionSize[1] = 3; // 设置通道数为 3
  valid_shape.dimensionSize[2] = image_height; // 设置图像高度
  valid_shape.dimensionSize[3] = image_width; // 设置图像宽度

  auto &aligned_shape = properties.alignedShape; // 获取对齐后的形状
  int32_t w_stride = ALIGN_16(image_width); // 计算对齐后的宽度
  aligned_shape.numDimensions = 4; // 设置维度数量为 4
  aligned_shape.dimensionSize[0] = 1; // 设置 batch size 为 1
  aligned_shape.dimensionSize[1] = 3; // 设置通道数为 3
  aligned_shape.dimensionSize[2] = image_height; // 设置图像高度
  aligned_shape.dimensionSize[3] = w_stride; // 设置图像宽度

  int32_t image_length = image_height * w_stride * 3 / 2; // 计算图像数据长度
  hbSysAllocCachedMem(&tensor->sysMem[0], image_length); // 分配缓存内存
}

LineFollowerPerceptionNode::LineFollowerPerceptionNode(const std::string &node_name, // 构造函数
                                                       const NodeOptions &options) // 节点选项
    : DnnNode(node_name, options) // 调用父类构造函数
{
  // 小车参数声明
  this->declare_parameter("model_path", "/userdata/dev_ws/src/originbot/originbot_deeplearning/line_follower_perception/model/resnet18_224x224_nv12_maze.bin"); // 声明模型路径参数
  this->declare_parameter("model_name", "resnet18_224x224_nv12"); 
  this->declare_parameter("go_time", 2.3); 
  this->declare_parameter("turning_vth", 2.0); 
  this->declare_parameter("data_path", "/userdata/dev_front/parse_res.txt"); 
  this->declare_parameter("straight_vth", 1.5); 
  this->declare_parameter("straight_turning_vth", 0.08); 

  
  if (GetParams() == false) 
  {
    RCLCPP_ERROR(this->get_logger(), "LineFollowerPerceptionNode GetParams() failed\n\n"); 
    return;
  }

  if (Init() != 0)
  {
    RCLCPP_ERROR(rclcpp::get_logger("LineFollowerPerceptionNode"), "Init failed!"); 
  }
  // 速度发布
  publisher_ =
      this->create_publisher<geometry_msgs::msg::Twist>("cmd_vel", 5);

  // 图片订阅
  subscriber_ =
      this->create_subscription<hbm_img_msgs::msg::HbmMsg1080P>(
          "hbmem_img",
          rclcpp::SensorDataQoS(),
          std::bind(&LineFollowerPerceptionNode::subscription_callback,
                    this,
                    std::placeholders::_1));

  // 小车停止信号订阅
  stop_sub_ = 
      this->create_subscription<std_msgs::msg::Bool>("/stop_car", 10,
          std::bind(&LineFollowerPerceptionNode::stopCallback, this, std::placeholders::_1));
  
  stop_car_ = false;
  IsTurn = false;     
  IsGo = false;       

  // 里程计订阅
  odom_subscription_ = 
      this->create_subscription<nav_msgs::msg::Odometry>("odom", 10, 
          std::bind(&odom_callback, std::placeholders::_1)); 
          
  std::ifstream file(data_path_); // 替换为您的文件名 // 打开文件
  if (!file.is_open()) // 如果文件打开失败
  {
    std::cerr << "Failed to open file" << std::endl; // 打印错误信息
  }
  std::string line;
  while (std::getline(file, line)) // 逐行读取文件
  {
    std::istringstream iss(line); // 创建字符串流
    char delimiter = '-'; // 定义分隔符
    std::string before, after; // 定义字符串
    // 读取"-"前的部分
    if (std::getline(iss, before, delimiter)) // 读取"-"前的部分
    {
      // 读取"-"后的部分
      std::getline(iss, after); // 读取"-"后的部分
      // 移除前后空格
      before.erase(0, before.find_first_not_of(" \t\n\r\f\v")); // 移除前空格
      before.erase(before.find_last_not_of(" \t\n\r\f\v") + 1); // 移除后空格
      after.erase(0, after.find_first_not_of(" \t\n\r\f\v")); // 移除前空格
      after.erase(after.find_last_not_of(" \t\n\r\f\v") + 1); // 移除后空格
      stringQueue.push(after); // 添加到队列
    }
  }
  //
}

LineFollowerPerceptionNode::~LineFollowerPerceptionNode() // 析构函数
{
}

bool LineFollowerPerceptionNode::GetParams() // 获取参数
{
  auto parameters_client = // 创建参数客户端
      std::make_shared<rclcpp::SyncParametersClient>(this);
  auto parameters = parameters_client->get_parameters( // 获取参数
      {"model_path", "model_name", "go_time", "turning_vth", "data_path", "straight_vth", "straight_turning_vth"});
  return AssignParams(parameters); // 赋值参数
}

bool LineFollowerPerceptionNode::AssignParams( // 赋值参数
    const std::vector<rclcpp::Parameter> &parameters) // 参数
{
  for (auto &parameter : parameters) // 遍历参数
  {
    if (parameter.get_name() == "model_path") // 如果参数名为 model_path
    {
      model_path_ = parameter.value_to_string(); // 赋值模型路径
    }
    else if (parameter.get_name() == "model_name") // 如果参数名为 model_name
    {
      model_name_ = parameter.value_to_string(); // 赋值模型名称
    }
    else if (parameter.get_name() == "go_time") // 如果参数名为 go_time
    {
      go_time_ = parameter.as_double(); // 赋值前进时间
    }
    else if (parameter.get_name() == "turning_vth") // 如果参数名为 turning_vth
    {
      turning_vth_ = parameter.as_double(); // 赋值转弯速度阈值
    }
    else if (parameter.get_name() == "data_path") // 如果参数名为 data_path
    {
      data_path_ = parameter.value_to_string(); // 赋值数据路径
    }
    else if (parameter.get_name() == "straight_vth") // 如果参数名为 data_path
    {
      straight_vth_ = parameter.as_double(); // 赋值数据路径
    }
    else if (parameter.get_name() == "straight_turning_vth") // 如果参数名为 data_path
    {
      straight_turning_vth_ = parameter.as_double(); // 赋值数据路径
    }
    else // 如果参数名无效
    {
      RCLCPP_WARN(this->get_logger(), "Invalid parameter name: %s", // 打印警告信息
                  parameter.get_name().c_str());
    }
  }
  RCLCPP_INFO(this->get_logger(), "model_path: %s", model_path_.c_str()); // 打印模型路径
  RCLCPP_INFO(this->get_logger(), "model_name: %s", model_name_.c_str()); // 打印模型名称
  RCLCPP_INFO(this->get_logger(), "go_time: %f", go_time_); // 打印前进时间
  RCLCPP_INFO(this->get_logger(), "turning_vth: %f", turning_vth_); // 打印转弯速度阈值
  RCLCPP_INFO(this->get_logger(), "data_path: %s", data_path_.c_str()); // 打印数据路径
  RCLCPP_INFO(this->get_logger(), "straight_vth: %f", straight_vth_); // 打印数据路径
  RCLCPP_INFO(this->get_logger(), "straight_turning_vth: %f", straight_turning_vth_); // 打印数据路径
  return true;
}

int LineFollowerPerceptionNode::SetNodePara() // 设置节点参数
{
  if (!dnn_node_para_ptr_) // 如果 dnn_node_para_ptr_ 为空
  {
    return -1;
  }
  RCLCPP_INFO(rclcpp::get_logger("LineFollowerPerceptionNode"), "path:%s\n", model_path_.c_str()); // 打印模型路径
  RCLCPP_INFO(rclcpp::get_logger("LineFollowerPerceptionNode"), "name:%s\n", model_name_.c_str()); // 打印模型名称
  dnn_node_para_ptr_->model_file = model_path_; // 赋值模型文件
  dnn_node_para_ptr_->model_name = model_name_; // 赋值模型名称
  dnn_node_para_ptr_->model_task_type = model_task_type_; // 赋值模型任务类型
  dnn_node_para_ptr_->task_num = 4; // 赋值任务数量
  return 0;
}

// 函数功能：把弧度限制在-PI~PI
double arcLimit(double arc)
{
  if (arc < -M_PI) // 如果弧度小于 -PI
    arc += 2 * M_PI; // 加上 2PI
  else if (arc > M_PI) // 如果弧度大于 PI
    arc -= 2 * M_PI; // 减去 2PI
  return arc; // 返回弧度
}

// 函数功能：寻找两个角度相差的最小角度
double findInferiorArc(double angle1, double angle2)
{
  angle1 = arcLimit(angle1); // 限制弧度
  angle2 = arcLimit(angle2); // 限制弧度
  double diff; // 差值
  if (angle1 > angle2) // 如果 angle1 大于 angle2
  {
    diff = angle1 - angle2; // 计算差值
  }
  else // 否则
  {
    diff = angle2 - angle1; // 计算差值
  }
  if (diff > M_PI) // 如果差值大于 PI
    diff = 2 * M_PI - diff; // 计算差值
  return diff; // 返回差值
}

// DNN-Node的后处理部分
int LineFollowerPerceptionNode::PostProcess( 
    const std::shared_ptr<DnnNodeOutput> &outputs) // 输出
{

  std::shared_ptr<LineCoordinateParser> line_coordinate_parser = // 创建 LineCoordinateParser 对象
      std::make_shared<LineCoordinateParser>();

  std::shared_ptr<LineCoordinateResult> result = // 创建 LineCoordinateResult 对象
      std::make_shared<LineCoordinateResult>();

  line_coordinate_parser->Parse(result, outputs->output_tensors[0]); // 解析结果

  int x = result->x;
  int y = result->y;

  // 打印坐标信息
  RCLCPP_INFO(rclcpp::get_logger("LineFollowerPerceptionNode"),
  "post coor x: %d    y:%d", x, y); 

  auto message = geometry_msgs::msg::Twist(); // 创建 Twist 消息
  // 将不需要的速度参数设置为0
  message.linear.y = 0.0;
  message.linear.z = 0.0;
  message.angular.x = 0.0;
  message.angular.y = 0.0;

  /*运动控制部分*/
  // 主控参数：
  // IsGo：为驶入岔路口提供一段直行时间，可调整IsGo里的GoedTime和线速度来调节岔路口驶入的准确度
  // IsTurn：处于转弯状态，可调整IsTurn里的角速度和线速度来调节转弯过程的速度
  // 目标控制参数：
  // 线速度（m/s）：message.linear.x 
  // 角速度（rad/s）：message.angular.z 

  if (stop_car_ == false)
  {
    // 识别到岔路口时
    if (y < 240 && IsTurn == false && IsGo == false) 
    { 
      IsGo = true;                  // 设置为前进
      message.linear.x = straight_vth_;      // 设置线速度（m/s）
      message.angular.z = 0;        // 设置角速度（rad/s）
      publisher_->publish(message); // 发布消息
    }
    else if (IsGo == true) 
    { 
      // 如果处于前进状态
      GoedTime += 0.035;            // 增加前进时间
      message.linear.x = straight_vth_;      // 设置线速度
      message.angular.z = 0;        // 设置角速度
      if (GoedTime > go_time_)      
      { 
        // 直行时间到达设定时间 判断左转、右转或直行
        if (stringQueue.empty()) 
        {
          stop_car_ = true;
          RCLCPP_INFO(rclcpp::get_logger("LineFollowerPerceptionNode"),
          "Action finish, car stop"); // 打印坐标信息
        }
        else
        {
          IsGo = false;               // 设置为不前进
          GoedTime = 0;               // 重置前进时间
          if (std::stoi(stringQueue.front()) == DIRECTION_LEFT) // 左转
          { 
            rpy[1].yaw = arcLimit(rpy[0].yaw + 1.57);     // 偏航角加 PI/2
            message.linear.x = straight_turning_vth_;            // 设置线速度
            message.angular.z = turning_vth_;   // 设置角速度
            IsTurn = true;                      // 设置为转弯状态
          }
          else if (std::stoi(stringQueue.front()) == DIRECTION_RIGHT) // 右转
          { 
            rpy[1].yaw = arcLimit(rpy[0].yaw - 1.57);     // 偏航角减 PI/2
            message.linear.x = straight_turning_vth_;             
            message.angular.z = -turning_vth_;  
            IsTurn = true;                      
          }
          else if (std::stoi(stringQueue.front()) == DIRECTION_BACK)
          {
            rpy[1].yaw = arcLimit(rpy[0].yaw - 3.14);     // 偏航角减 PI
            message.linear.x = 0.0;                       
            message.angular.z = -turning_vth_;           
            IsTurn = true;                                
          }
          else if (std::stoi(stringQueue.front()) == DIRECTION_FRONT)// 直行
          { 
            stringQueue.pop(); // 直接弹出队列
          }
        }
      }
      publisher_->publish(message); // 发布消息
    }
    else if (IsTurn == true) // 处于转弯状态
    { 
      //打印调试信息
      RCLCPP_INFO(rclcpp::get_logger("LineFollowerPerceptionNode"),
      "Turning... current_yaw: %.3f, target_yaw: %.3f, diff: %.3f",
      rpy[0].yaw, rpy[1].yaw, findInferiorArc(rpy[1].yaw, rpy[0].yaw));
      if (stringQueue.empty())
      {
        stop_car_ = true;
        RCLCPP_INFO(rclcpp::get_logger("LineFollowerPerceptionNode"),
        "Action finish, car stop"); // 打印坐标信息
      }
      else if (std::stoi(stringQueue.front()) == DIRECTION_LEFT) // 左转
      {
        message.angular.z = turning_vth_;   // 设置角速度
        message.linear.x = straight_turning_vth_; // 设置线速度
      }
      else if (std::stoi(stringQueue.front()) == DIRECTION_RIGHT)// 右转
      {
        message.angular.z = -turning_vth_;  // 设置角速度
        message.linear.x = straight_turning_vth_; // 设置线速度
      }
      else if (std::stoi(stringQueue.front()) == DIRECTION_BACK)//转弯
      {
        message.angular.z = -turning_vth_;  // 设置角速度
        message.linear.x = 0.0; // 设置线速度
      }
      if (findInferiorArc(rpy[1].yaw, rpy[0].yaw) < 0.1) // 如果转弯结束
      { 
        IsTurn = false;             // 设置为不转弯
        stringQueue.pop();          // 弹出队列
        message.angular.z = 0.0;    // 设置角速度
      }
      publisher_->publish(message); // 发布消息
    }
    else // 循线
    { 
      float angular_z = -1.0 * (x - 320) / 200.0; // 计算角速度
      message.linear.x = straight_vth_;                    // 设置线速度
      message.angular.z = angular_z;              // 设置角速度
      publisher_->publish(message);               // 发布消息
    }
  }
  if (stop_car_) 
  {
    message.linear.x = 0.0;
    message.angular.z = 0.0;
    publisher_->publish(message);
  }
  return 0;
}

// 里程计回调函数
void odom_callback( 
    const nav_msgs::msg::Odometry::SharedPtr msg) // 消息
{
  if (!msg || !rclcpp::ok()) // 如果消息为空或者 rclcpp 未初始化
  {
    return;
  }
  tf2::Quaternion q(msg->pose.pose.orientation.x, 
                    msg->pose.pose.orientation.y, 
                    msg->pose.pose.orientation.z, 
                    msg->pose.pose.orientation.w); 
  double roll, pitch, yaw; // 定义翻滚角，俯仰角，偏航角
  tf2::Matrix3x3 m(q); // 创建矩阵
  m.getRPY(roll, pitch, yaw); // 获取翻滚角，俯仰角，偏航角
  rpy[0].roll = roll; // 赋值翻滚角
  rpy[0].pitch = pitch; // 赋值俯仰角
  rpy[0].yaw = yaw; // 赋值偏航角
}


// 小车停止回调函数
void LineFollowerPerceptionNode::stopCallback(const std_msgs::msg::Bool::SharedPtr msg)
{
  stop_car_ = msg->data;
  if (stop_car_) {
    RCLCPP_INFO(this->get_logger(), "Stop command received.");
  }
}

// 订阅回调函数
void LineFollowerPerceptionNode::subscription_callback( 
    const hbm_img_msgs::msg::HbmMsg1080P::SharedPtr msg) // 消息
{

  if (!msg || !rclcpp::ok()) // 如果消息为空或者 rclcpp 未初始化
  {
    return;
  }
  std::stringstream ss; // 创建字符串流
  ss << "Recved img encoding: " // 拼接字符串
     << std::string(reinterpret_cast<const char *>(msg->encoding.data()))
     << ", h: " << msg->height << ", w: " << msg->width
     << ", step: " << msg->step << ", index: " << msg->index
     << ", stamp: " << msg->time_stamp.sec << "_"
     << msg->time_stamp.nanosec << ", data size: " << msg->data_size;
  RCLCPP_DEBUG(rclcpp::get_logger("LineFollowerPerceptionNode"), "%s", ss.str().c_str()); // 打印调试信息

  auto model_manage = GetModel(); // 获取模型
  if (!model_manage) // 如果模型为空
  {
    RCLCPP_ERROR(rclcpp::get_logger("LineFollowerPerceptionNode"), "Invalid model"); // 打印错误信息
    return;
  }

  hbDNNRoi roi; // 创建 hbDNNRoi 对象
  roi.left = 0; // 设置左边界
  roi.top = 128; // 设置上边界
  roi.right = 640; // 设置右边界
  roi.bottom = 352; // 设置下边界

  cv::Mat img_mat(msg->height * 3 / 2, msg->width, CV_8UC1, (void *)(msg->data.data())); // 创建 cv::Mat 对象
  cv::Range rowRange(roi.top, roi.bottom); // 创建 cv::Range 对象
  cv::Range colRange(roi.left, roi.right); // 创建 cv::Range 对象
  cv::Mat crop_img_mat = hobot_cv::hobotcv_crop(img_mat, msg->height, msg->width, 224, 224, rowRange, colRange); // 裁剪图像

  std::shared_ptr<hobot::dnn_node::NV12PyramidInput> pyramid = nullptr; // 创建 NV12PyramidInput 对象
  pyramid = hobot::dnn_node::ImageProc::GetNV12PyramidFromNV12Img( // 获取 NV12 金字塔图像
      reinterpret_cast<const char *>(crop_img_mat.data),
      224,
      224,
      224,
      224);
  if (!pyramid) // 如果金字塔图像为空
  {
    RCLCPP_ERROR(rclcpp::get_logger("LineFollowerPerceptionNode"), "Get Nv12 pym fail!"); // 打印错误信息
    return;
  }
  std::vector<std::shared_ptr<DNNInput>> inputs; // 创建 DNNInput 向量
  auto rois = std::make_shared<std::vector<hbDNNRoi>>(); // 创建 hbDNNRoi 向量
  roi.left = 0; // 设置左边界
  roi.top = 0; // 设置上边界
  roi.right = 224; // 设置右边界
  roi.bottom = 224; // 设置下边界
  rois->push_back(roi); // 添加 roi

  for (size_t i = 0; i < rois->size(); i++) // 遍历 rois
  {
    for (int32_t j = 0; j < model_manage->GetInputCount(); j++) // 遍历输入
    {
      inputs.push_back(pyramid); // 添加输入
    }
  }

  auto dnn_output = std::shared_ptr<DnnNodeOutput>(); // 创建 DnnNodeOutput 对象

  Predict(inputs, dnn_output, rois); // 预测
}

int LineFollowerPerceptionNode::Predict( // 预测
    std::vector<std::shared_ptr<DNNInput>> &dnn_inputs, // 输入
    const std::shared_ptr<DnnNodeOutput> &output, // 输出
    const std::shared_ptr<std::vector<hbDNNRoi>> rois) // roi
{
  // RCLCPP_INFO(rclcpp::get_logger("LineFollowerPerceptionNode"), "input size:%ld roi size:%ld", dnn_inputs.size(), rois->size()); // 打印输入和 roi 大小
  return Run(dnn_inputs,
             output,
             rois,
             true);
}

int32_t LineCoordinateParser::Parse( // 解析线条坐标
    std::shared_ptr<LineCoordinateResult> &output, // 输出
    std::shared_ptr<DNNTensor> &output_tensor) // 输出 tensor
{
  if (!output_tensor) // 如果输出 tensor 为空
  {
    RCLCPP_ERROR(rclcpp::get_logger("LineFollowerPerceptionNode"), "invalid out tensor"); // 打印错误信息
    rclcpp::shutdown(); // 关闭 rclcpp
  }
  std::shared_ptr<LineCoordinateResult> result; // 创建 LineCoordinateResult 对象
  if (!output) // 如果输出为空
  {
    result = std::make_shared<LineCoordinateResult>(); // 创建 LineCoordinateResult 对象
    output = result; // 赋值输出
  }
  else // 否则
  {
    result = std::dynamic_pointer_cast<LineCoordinateResult>(output); // 转换类型
  }
  DNNTensor &tensor = *output_tensor; // 获取 tensor
  const int32_t *shape = tensor.properties.validShape.dimensionSize; // 获取形状
  RCLCPP_DEBUG(rclcpp::get_logger("LineFollowerPerceptionNode"),
               "PostProcess shape[1]: %d shape[2]: %d shape[3]: %d",
               shape[1],
               shape[2],
               shape[3]); // 打印形状信息
  hbSysFlushMem(&(tensor.sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE); // 刷新缓存
  float x = reinterpret_cast<float *>(tensor.sysMem[0].virAddr)[0]; // 获取 x 坐标
  float y = reinterpret_cast<float *>(tensor.sysMem[0].virAddr)[1]; // 获取 y 坐标
  result->x = (x * 112 + 112) * 640.0 / 224.0; // 计算 x 坐标
  result->y = 224 - (y * 112 + 112) + 240; // 计算 y 坐标
  // RCLCPP_INFO(rclcpp::get_logger("LineFollowerPerceptionNode"),
  //             "coor rawx: %f,  rawy:%f, x: %f    y:%f", x, y, result->x, result->y); // 打印坐标信息
  return 0;
}

int main(int argc, char *argv[]) // 主函数
{

  rclcpp::init(argc, argv); // 初始化 rclcpp
  RCLCPP_INFO(rclcpp::get_logger("LineFollowerPerceptionNode"), "Pkg start."); // 打印信息
  rclcpp::spin(std::make_shared<LineFollowerPerceptionNode>("GetLineCoordinate")); // 运行节点

  rclcpp::shutdown(); // 关闭 rclcpp

  RCLCPP_WARN(rclcpp::get_logger("LineFollowerPerceptionNode"), "Pkg exit."); // 打印警告信息
  return 0;
}
