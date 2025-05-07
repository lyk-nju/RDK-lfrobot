import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import GetParameters, SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from transforms3d.euler import quat2euler
import gradio as gr
import threading
import math
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image

class CarStatusVisualizer(Node):
    def __init__(self):
        super().__init__('car_status_visualizer')
        self.remote_node_name = 'line_follower_perception_node'
        self.parameter_names = ['turning_vth', 'straight_turning_vth', 'straight_vth', 'go_time']

        self.stop_car = False
        self.linear_x = 0.0
        self.angular_z = 0.0
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.yaw = 0.0
        self.trajectory = []

        self.create_subscription(Bool, '/stop_car', self.stop_car_callback, 10)
        self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

    def stop_car_callback(self, msg):
        self.stop_car = msg.data

    def cmd_vel_callback(self, msg):
        self.linear_x = msg.linear.x
        self.angular_z = msg.angular.z

    def odom_callback(self, msg):
        self.pos_x = msg.pose.pose.position.x
        self.pos_y = msg.pose.pose.position.y
        orientation_q = msg.pose.pose.orientation
        yaw = quat2euler([orientation_q.w, orientation_q.x, orientation_q.y, orientation_q.z])[2]
        self.yaw = -yaw
        self.trajectory.append((self.pos_x, self.pos_y))
        if len(self.trajectory) > 1000:
            self.trajectory.pop(0)

    def generate_plot(self):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_title("Car Position and Yaw")

        if len(self.trajectory) > 1:
            traj_x, traj_y = zip(*self.trajectory)
            ax.plot(traj_x, traj_y, 'b-')

        ax.plot(self.pos_x, self.pos_y, 'ro')
        dx = math.cos(self.yaw)
        dy = math.sin(self.yaw)
        ax.arrow(self.pos_x, self.pos_y, dx, dy, head_width=0.3, head_length=0.5, fc='r', ec='r')

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return Image.open(buf)

    def get_remote_parameters(self):
        client = self.create_client(GetParameters, f'{self.remote_node_name}/get_parameters')
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for remote parameter service...')
        request = GetParameters.Request()
        request.names = self.parameter_names
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        values = [p.double_value for p in future.result().values]
        return values

    def set_remote_parameters(self, turning_vth, straight_turning_vth, straight_vth, go_time):
        client = self.create_client(SetParameters, f'{self.remote_node_name}/set_parameters')
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for remote set_parameters service...')
        request = SetParameters.Request()
        request.parameters = [
            Parameter(name='turning_vth', value=ParameterValue(type=3, double_value=turning_vth)),
            Parameter(name='straight_turning_vth', value=ParameterValue(type=3, double_value=straight_turning_vth)),
            Parameter(name='straight_vth', value=ParameterValue(type=3, double_value=straight_vth)),
            Parameter(name='go_time', value=ParameterValue(type=3, double_value=go_time)),
        ]
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

    def get_status(self):
        try:
            turning_vth, straight_turning_vth, straight_vth, go_time = self.get_remote_parameters()
        except Exception as e:
            self.get_logger().error(f"Failed to get remote parameters: {e}")
            turning_vth = straight_turning_vth = straight_vth = go_time = -1.0

        is_turn = abs(self.angular_z) > 0
        is_go = abs(self.linear_x) > 0 and not is_turn

        return {
            "turning_vth": turning_vth,
            "straight_turning_vth": straight_turning_vth,
            "straight_vth": straight_vth,
            "go_time": go_time,
            "stop_car": self.stop_car,
            "cmd_vel.linear.x": self.linear_x,
            "cmd_vel.angular.z": self.angular_z,
            "IsGo": is_go,
            "IsTurn": is_turn,
            "pos_x": self.pos_x,
            "pos_y": self.pos_y,
            "yaw": self.yaw,
            "plot": self.generate_plot()
        }

def start_ros_node(node):
    rclpy.spin(node)

def launch_gradio_ui(node):
    def get_full_state():
        status = node.get_status()
        status_color = "red" if status["stop_car"] else "green"
        html_lines = []
        if status["stop_car"]:
            html_lines.append("<span style='color:red;'>StopCar: True</span>")
        if status["IsTurn"]:
            html_lines.append("<span style='color:orange;'>IsTurn: True</span>")
        if status["IsGo"]:
            html_lines.append("<span style='color:green;'>IsGo: True</span>")

        if not html_lines:
            html_lines.append("<span style='color:gray;'>No active state</span>")

        html_str = "<div style='font-size: 18px;'>" + "<br>".join(html_lines) + "</div>"
        return (
            status["turning_vth"], status["straight_turning_vth"],
            status["straight_vth"], status["go_time"],
            html_str,
            status["cmd_vel.linear.x"], status["cmd_vel.angular.z"],
            status["pos_x"], status["pos_y"], status["yaw"],
            status["plot"]
        )

    def get_status_only():
        return get_full_state()

    def apply_params(turning_vth_val, straight_turning_vth_val, straight_vth_val, go_time_val):
        node.set_remote_parameters(turning_vth_val, straight_turning_vth_val, straight_vth_val, go_time_val)

    with gr.Blocks() as demo:
        gr.Markdown("<h1 style='text-align: center;'>🚗 Originbot Car</h1>")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🪛 运动参数")
                turning_vth = gr.Number(label="Turning Vth", step=0.1)
                straight_turning_vth = gr.Number(label="Straight Turning Vth", step=0.1)
                straight_vth = gr.Number(label="Straight Vth", step=0.1)
                go_time = gr.Number(label="Go Time", step=0.1)

                apply_btn = gr.Button("✅ 应用参数")
                update_btn = gr.Button("🔄 手动刷新")

            with gr.Column():
                gr.Markdown("### 🟢 运行状态")
                status_box = gr.HTML(label="状态显示")

                gr.Markdown("### 📈 位姿状态")
                linear_x = gr.Number(label="Linear X")
                angular_z = gr.Number(label="Angular Z")
                pos_x = gr.Number(label="Pos X")
                pos_y = gr.Number(label="Pos Y")
                yaw = gr.Number(label="Yaw")
                plot = gr.Image(label="轨迹图")

        update_btn.click(
            fn=get_full_state,
            inputs=[],
            outputs=[turning_vth, straight_turning_vth, straight_vth, go_time,
                     status_box, linear_x, angular_z, pos_x, pos_y, yaw, plot]
        )

        apply_btn.click(
            fn=apply_params,
            inputs=[turning_vth, straight_turning_vth, straight_vth, go_time],
            outputs=[]
        )

        timer = gr.Timer(0.2)
        timer.tick(
            fn=get_status_only,
            inputs=[],
            outputs=[turning_vth, straight_turning_vth, straight_vth, go_time,
                     status_box, linear_x, angular_z, pos_x, pos_y, yaw, plot]
        )

        demo.launch(server_name="0.0.0.0", server_port=7861)

def main():
    rclpy.init()
    node = CarStatusVisualizer()
    t = threading.Thread(target=start_ros_node, args=(node,), daemon=True)
    t.start()
    launch_gradio_ui(node)

if __name__ == '__main__':
    main()
