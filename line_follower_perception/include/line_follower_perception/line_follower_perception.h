//  该文件定义了LineFollowerPerceptionNode类和LineCoordinateParser类
//  LineFollowerPerceptionNode类是巡线系统的核心组件，负责接收图像数据，使用深度学习模型识别线条位置，并根据线条位置控制机器人的运动。
//  LineCoordinateParser类包含一个Parse方法，用于解析模型输出，提取线条坐标。
// Copyright (c) 2024，D-Robotics.
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

#ifndef _LINE_FOLLOWER_PERCEPTION_H_
#define _LINE_FOLLOWER_PERCEPTION_H_

#include <opencv2/opencv.hpp>

#include "rclcpp/rclcpp.hpp"
#include "dnn_node/dnn_node.h"
#include "dnn_node/dnn_node_data.h"
#include "hbm_img_msgs/msg/hbm_msg1080_p.hpp"
#include "std_msgs/msg/int16_multi_array.hpp"
#include "geometry_msgs/msg/twist.hpp"


using rclcpp::NodeOptions;

using hobot::dnn_node::DNNInput;
using hobot::dnn_node::DnnNode;
using hobot::dnn_node::DnnNodeOutput;
using hobot::dnn_node::ModelTaskType;

using hobot::dnn_node::DNNTensor;

class LineCoordinateResult {
 public:
  float x;
  float y;
  void Reset() {x = -1.0; y = -1.0;}
};

class LineCoordinateParser {
 public:
  LineCoordinateParser() {}
  ~LineCoordinateParser() {}
  int32_t Parse(
      std::shared_ptr<LineCoordinateResult>& output,
      std::shared_ptr<DNNTensor>& output_tensor);
};

class LineFollowerPerceptionNode : public DnnNode {
 public:
  LineFollowerPerceptionNode(const std::string& node_name,
                        const NodeOptions &options = NodeOptions());
  ~LineFollowerPerceptionNode() override;

 protected:
  int SetNodePara() override;
  int PostProcess(const std::shared_ptr<DnnNodeOutput> &outputs) override;

 private:
  int Predict(std::vector<std::shared_ptr<DNNInput>> &dnn_inputs,
              const std::shared_ptr<DnnNodeOutput> &output,
              const std::shared_ptr<std::vector<hbDNNRoi>> rois);
  void subscription_callback(const hbm_img_msgs::msg::HbmMsg1080P::SharedPtr msg);
  void stopCallback(const std_msgs::msg::Bool::SharedPtr msg);
  bool GetParams();
  bool AssignParams(const std::vector<rclcpp::Parameter> & parameters);
  ModelTaskType model_task_type_ = ModelTaskType::ModelInferType;

  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr stop_sub_;
  rclcpp::Subscription<hbm_img_msgs::msg::HbmMsg1080P>::SharedPtr subscriber_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr publisher_;

  cv::Mat image_bgr_;
  std::string model_path_;
  std::string model_name_;
};

#endif  // _LINE_FOLLOWER_PERCEPTION_H_
