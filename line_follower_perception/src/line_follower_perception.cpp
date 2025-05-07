//  该文件是line_follower_perception节点的源代码文件，其中包含了节点的所有功能的具体实现。
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

#include "line_follower_perception/line_follower_perception.h"  // 包含头文件

#include <fstream>   // 包含文件流头文件
#include <string>    // 包含字符串头文件
#include <opencv2/opencv.hpp>  // 包含 OpenCV 头文件

#include "dnn_node/util/image_proc.h"  // 包含图像处理工具头文件
#include "hobot_cv/hobotcv_imgproc.h"  // 包含地平线 CV 图像处理头文件

// 准备 NV12 格式的 tensor，不进行 padding
void prepare_nv12_tensor_without_padding(const char *image_data,
                                         int image_height,
                                         int image_width,
                                         hbDNNTensor *tensor) {
  auto &properties = tensor->properties;  // 获取 tensor 的属性
  properties.tensorType = HB_DNN_IMG_TYPE_NV12;  // 设置 tensor 类型为 NV12 图像
  properties.tensorLayout = HB_DNN_LAYOUT_NCHW;  // 设置 tensor 布局为 NCHW
  auto &valid_shape = properties.validShape;  // 获取有效形状
  valid_shape.numDimensions = 4;  // 设置维度数量为 4
  valid_shape.dimensionSize[0] = 1;  // 设置 batch size 为 1
  valid_shape.dimensionSize[1] = 3;  // 设置通道数为 3
  valid_shape.dimensionSize[2] = image_height;  // 设置图像高度
  valid_shape.dimensionSize[3] = image_width;  // 设置图像宽度

  auto &aligned_shape = properties.alignedShape;  // 获取对齐后的形状
  aligned_shape = valid_shape;  // 对齐后的形状与有效形状相同

  int32_t image_length = image_height * image_width * 3 / 2;  // 计算图像数据长度

  hbSysAllocCachedMem(&tensor->sysMem[0], image_length);  // 分配缓存内存
  memcpy(tensor->sysMem[0].virAddr, image_data, image_length);  // 拷贝图像数据

  hbSysFlushMem(&(tensor->sysMem[0]), HB_SYS_MEM_CACHE_CLEAN);  // 刷新缓存
}

// 准备 NV12 格式的 tensor，不进行 padding (重载版本)
void prepare_nv12_tensor_without_padding(int image_height,
                                         int image_width,
                                         hbDNNTensor *tensor) {
  auto &properties = tensor->properties;  // 获取 tensor 的属性
  properties.tensorType = HB_DNN_IMG_TYPE_NV12;  // 设置 tensor 类型为 NV12 图像
  properties.tensorLayout = HB_DNN_LAYOUT_NCHW;  // 设置 tensor 布局为 NCHW

  auto &valid_shape = properties.validShape;  // 获取有效形状
  valid_shape.numDimensions = 4;  // 设置维度数量为 4
  valid_shape.dimensionSize[0] = 1;  // 设置 batch size 为 1
  valid_shape.dimensionSize[1] = 3;  // 设置通道数为 3
  valid_shape.dimensionSize[2] = image_height;  // 设置图像高度
  valid_shape.dimensionSize[3] = image_width;  // 设置图像宽度

  auto &aligned_shape = properties.alignedShape;  // 获取对齐后的形状
  int32_t w_stride = ALIGN_16(image_width);  // 计算对齐后的宽度
  aligned_shape.numDimensions = 4;  // 设置维度数量为 4
  aligned_shape.dimensionSize[0] = 1;  // 设置 batch size 为 1
  aligned_shape.dimensionSize[1] = 3;  // 设置通道数为 3
  aligned_shape.dimensionSize[2] = image_height;  // 设置图像高度
  aligned_shape.dimensionSize[3] = w_stride;  // 设置对齐后的图像宽度

  int32_t image_length = image_height * w_stride * 3 / 2;  // 计算图像数据长度
  hbSysAllocCachedMem(&tensor->sysMem[0], image_length);  // 分配缓存内存
}

// 构造函数，初始化 LineFollowerPerceptionNode 节点
LineFollowerPerceptionNode::LineFollowerPerceptionNode(const std::string& node_name,
                      const NodeOptions& options)
  : DnnNode(node_name, options) {
  this->declare_parameter("model_path", "./resnet18_224x224_nv12.bin"); // 声明模型路径参数
  this->declare_parameter("model_name", "resnet18_224x224_nv12");  // 声明模型名称参数
  if (GetParams() == false) { // 获取参数
    RCLCPP_ERROR(this->get_logger(), "LineFollowerPerceptionNode GetParams() failed\n\n"); // 打印错误信息
    return;
  }
  
  if (Init() != 0) { // 初始化
    RCLCPP_ERROR(rclcpp::get_logger("LineFollowerPerceptionNode"), "Init failed!"); // 打印错误信息
  }

  publisher_ = // 创建发布者
    this->create_publisher<geometry_msgs::msg::Twist>("cmd_vel", 5);
  subscriber_ =  // 创建订阅者
    this->create_subscription<hbm_img_msgs::msg::HbmMsg1080P>(
      "hbmem_img",
      rclcpp::SensorDataQoS(),
      std::bind(&LineFollowerPerceptionNode::subscription_callback,
      this,
      std::placeholders::_1)); 
}

LineFollowerPerceptionNode::~LineFollowerPerceptionNode() { // 析构函数

}

bool LineFollowerPerceptionNode::GetParams() { // 获取参数
  auto parameters_client = // 创建参数客户端
    std::make_shared<rclcpp::SyncParametersClient>(this);
  auto parameters = parameters_client->get_parameters( // 获取参数
    {"model_path", "model_name"});
  return AssignParams(parameters); // 赋值参数
}

bool LineFollowerPerceptionNode::AssignParams( // 赋值参数
  const std::vector<rclcpp::Parameter> & parameters) {
  for (auto & parameter : parameters) { // 遍历参数
    if (parameter.get_name() == "model_path") { // 如果参数名为 model_path
      model_path_ = parameter.value_to_string(); // 赋值模型路径
    } else if (parameter.get_name() == "model_name") { // 如果参数名为 model_name
      model_name_ = parameter.value_to_string(); // 赋值模型名称
    } else {
      RCLCPP_WARN(this->get_logger(), "Invalid parameter name: %s", // 打印警告信息
        parameter.get_name().c_str());
    }
  }
  return true;
}

int LineFollowerPerceptionNode::SetNodePara() { // 设置节点参数
  if (!dnn_node_para_ptr_) { // 如果 dnn_node_para_ptr_ 为空
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

int LineFollowerPerceptionNode::PostProcess( // 后处理
  const std::shared_ptr<DnnNodeOutput> &outputs) {

  std::shared_ptr<LineCoordinateParser> line_coordinate_parser = // 创建 LineCoordinateParser 对象
      std::make_shared<LineCoordinateParser>();

  std::shared_ptr<LineCoordinateResult> result = // 创建 LineCoordinateResult 对象
      std::make_shared<LineCoordinateResult>();

  line_coordinate_parser->Parse(result, outputs->output_tensors[0]); // 解析结果

  int x = result->x; // 获取 x 坐标
  int y = result->y; // 获取 y 坐标
  RCLCPP_INFO(rclcpp::get_logger("LineFollowerPerceptionNode"),
               "post coor x: %d    y:%d", x, y); // 打印坐标信息
  float angular_z = -1.0 * (x - 320) / 200.0; // 计算角速度
  auto message = geometry_msgs::msg::Twist(); // 创建 Twist 消息
  message.linear.x = 0.1; // 设置线速度
  message.linear.y = 0.0; // 设置线速度
  message.linear.z = 0.0; // 设置线速度
  message.angular.x = 0.0; // 设置角速度
  message.angular.y = 0.0; // 设置角速度
  message.angular.z = angular_z; // 设置角速度
  publisher_->publish(message); // 发布消息
  return 0;
}

void LineFollowerPerceptionNode::subscription_callback( // 订阅回调函数
    const hbm_img_msgs::msg::HbmMsg1080P::SharedPtr msg) {
      
  if (!msg || !rclcpp::ok()) { // 如果消息为空或者 rclcpp 未初始化
    return;
  }
  std::stringstream ss; // 创建字符串流
  ss << "Recved img encoding: " // 拼接字符串
     << std::string(reinterpret_cast<const char*>(msg->encoding.data()))
     << ", h: " << msg->height << ", w: " << msg->width
     << ", step: " << msg->step << ", index: " << msg->index
     << ", stamp: " << msg->time_stamp.sec << "_"
     << msg->time_stamp.nanosec << ", data size: " << msg->data_size;
  RCLCPP_DEBUG(rclcpp::get_logger("LineFollowerPerceptionNode"), "%s", ss.str().c_str()); // 打印调试信息

  auto model_manage = GetModel(); // 获取模型
  if (!model_manage) { // 如果模型为空
    RCLCPP_ERROR(rclcpp::get_logger("LineFollowerPerceptionNode"), "Invalid model"); // 打印错误信息
    return;
  }

  hbDNNRoi roi; // 创建 hbDNNRoi 对象
  roi.left = 0; // 设置左边界
  roi.top = 128; // 设置上边界
  roi.right = 640; // 设置右边界
  roi.bottom = 352; // 设置下边界

  cv::Mat img_mat(msg->height * 3 / 2, msg->width, CV_8UC1, (void*)(msg->data.data())); // 创建 cv::Mat 对象
  cv::Range rowRange(roi.top, roi.bottom); // 创建 cv::Range 对象
  cv::Range colRange(roi.left, roi.right); // 创建 cv::Range 对象
  cv::Mat crop_img_mat = hobot_cv::hobotcv_crop(img_mat, msg->height, msg->width, 224, 224, rowRange, colRange); // 裁剪图像

  std::shared_ptr<hobot::dnn_node::NV12PyramidInput> pyramid = nullptr; // 创建 NV12PyramidInput 对象
  pyramid = hobot::dnn_node::ImageProc::GetNV12PyramidFromNV12Img( // 获取 NV12 金字塔图像
      reinterpret_cast<const char*>(crop_img_mat.data),
      224,
      224,
      224,
      224);
  if (!pyramid) { // 如果金字塔图像为空
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

  for (size_t i = 0; i < rois->size(); i++) { // 遍历 rois
    for (int32_t j = 0; j < model_manage->GetInputCount(); j++) { // 遍历输入
      inputs.push_back(pyramid); // 添加输入
    }
  }

  auto dnn_output = std::shared_ptr<DnnNodeOutput>(); // 创建 DnnNodeOutput 对象

  Predict(inputs, dnn_output, rois); // 预测
}

int LineFollowerPerceptionNode::Predict( // 预测
  std::vector<std::shared_ptr<DNNInput>> &dnn_inputs,
  const std::shared_ptr<DnnNodeOutput> &output,
  const std::shared_ptr<std::vector<hbDNNRoi>> rois) {
  RCLCPP_INFO(rclcpp::get_logger("LineFollowerPerceptionNode"), "input size:%ld roi size:%ld", dnn_inputs.size(), rois->size()); // 打印输入和 roi 大小
  return Run(dnn_inputs,
             output,
             rois,
             true);
}

int32_t LineCoordinateParser::Parse( // 解析线条坐标
    std::shared_ptr<LineCoordinateResult> &output,
    std::shared_ptr<DNNTensor> &output_tensor) {
  if (!output_tensor) { // 如果输出 tensor 为空
    RCLCPP_ERROR(rclcpp::get_logger("LineFollowerPerceptionNode"), "invalid out tensor"); // 打印错误信息
    rclcpp::shutdown(); // 关闭 rclcpp
  }
  std::shared_ptr<LineCoordinateResult> result; // 创建 LineCoordinateResult 对象
  if (!output) { // 如果输出为空
    result = std::make_shared<LineCoordinateResult>(); // 创建 LineCoordinateResult 对象
    output = result; // 赋值输出
  } else {
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
  RCLCPP_INFO(rclcpp::get_logger("LineFollowerPerceptionNode"),
               "coor rawx: %f,  rawy:%f, x: %f    y:%f", x, y, result->x, result->y); // 打印坐标信息
  return 0;
}

int main(int argc, char* argv[]) { // 主函数

  rclcpp::init(argc, argv); // 初始化 rclcpp
  RCLCPP_INFO(rclcpp::get_logger("LineFollowerPerceptionNode"), "Pkg start."); // 打印信息
  rclcpp::spin(std::make_shared<LineFollowerPerceptionNode>("GetLineCoordinate")); // 运行节点

  rclcpp::shutdown(); // 关闭 rclcpp

  RCLCPP_WARN(rclcpp::get_logger("LineFollowerPerceptionNode"), "Pkg exit."); // 打印警告信息
  return 0;
}
