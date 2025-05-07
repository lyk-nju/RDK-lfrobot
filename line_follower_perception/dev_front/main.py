import os  
import threading  
import subprocess  
import gradio as gr  
import time  

from frontend.utils import llm_maze_infer, llm_infer_validate, llm_asr_infer  
from frontend import Autodriver  
from frontend.ASR import ASRClient

# === 配置 ===
APPID = "9fe217f8"
API_KEY = "0b6a94e0344907568eca1f09d9828c2c"
URL = "ws://rtasr.xfyun.cn/v1/ws"
api_key_ = "sk-137c8b0990634abd8cd7f5317f0a018fmmz8zx2eruwprsmt"

# === 延迟创建的 ASRClient 实例 ===
asr_client = None

def main():
    title = """<center><h1> Volcano LLM Maze Show 🛰️ </h1></center>"""

    def launch_ros():
        threading.Thread(target=exit_and_run_ros, daemon=True).start()
        return gr.update(visible=True)

    def exit_and_run_ros():
        try:
            print("Closing Gradio...")
            gr.close_all()
            demo.close()
        except Exception as e:
            print(f"Error while closing Gradio: {e}")
        try:
            print("Starting ROS process...")
            subprocess.run(
                ["ros2", "launch", "line_follower_perception", "line_follower_maze.launch.py"],
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Failed to launch ROS: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

    def get_turns_from_ui(cross1, cross2, cross3):
        cmd_dict = {"Left": "左转", "Right": "右转", "Straight": "直行"}
        path = [cmd_dict[cross1], cmd_dict[cross2], cmd_dict[cross3]]
        input_text = "，".join(path)
        res_str = llm_maze_infer(input_text, api_key_)
        try:
            with open("parse_res.txt", "w") as f:
                f.write(res_str)
                print(f"finish writing from UI: {res_str}")
        except Exception as e:
            print("Error creating parse_res.txt:", e)
        if llm_infer_validate("parse_res.txt"):
            return f"输入路径:\n{input_text}", gr.update(visible=True)
        else:
            return "解析失败", gr.update(visible=False)

    def start_recording_and_update_button():
        global asr_client
        if asr_client is None:
            print("Initializing ASRClient...")
            asr_client = ASRClient(appid=APPID, api_key=API_KEY, url=URL)
        asr_client.start_recording()
        return gr.update(value="🎤 录音中...", variant="primary")

    def stop_recording_and_update_button():
        global asr_client
        if asr_client is None:
            return "ASR 未初始化，请先点击开始录音", gr.update(visible=False)

        asr_client.stop_recording()
        time.sleep(2)
        input_text = asr_client.get_result()
        print("识别到的输入:", input_text)

        if not input_text:
            return "未识别到输入，请重新尝试。", gr.update(visible=False)
        
        try:
            res_str = llm_asr_infer(input_text, api_key_)
        except Exception as e:
            print(f"调用 llm_asr_infer 时出错: {e}")
            return "解析出错，请稍后重试。", gr.update(visible=False)

        try:
            with open("parse_res.txt", "w") as f:
                f.write(res_str)
                print(f"写入文件成功: {res_str}")
        except Exception as e:
            print("创建 parse_res.txt 文件时出错:", e)

        if llm_infer_validate("parse_res.txt"):
            return f"识别结果:{input_text}", gr.update(visible=True)
        else:
            return "解析失败", gr.update(visible=False)

    def plan_path_and_save(start, goals, start_angle):
        print(f"Start point: {start}, Goal point: {goals}")
        full_path, full_codes = Autodriver.plan_route(start, goals, start_angle)
        print(f"Full path: {full_path}")
        if full_path:
            try:
                with open("parse_res.txt", "w") as f:
                    for i, code in enumerate(full_codes, 1):
                        f.write(f"{i}-{code}\n")
                print("successfully save")
            except Exception as e:
                print(f"写入文件出错: {e}")
            return f"规划路径:\n{' -> '.join(full_path)}", gr.update(visible=True)
        else:
            return "路径规划失败", gr.update(visible=False)

    def stop_car():
        try:
            subprocess.run(
                ["ros2", "topic", "pub", "/stop_car", "std_msgs/msg/Bool", "{data: true}"],
                check=True
            )
            return "已发送停止命令"
        except subprocess.CalledProcessError as e:
            return f"发送停止命令失败: {e}"
        except Exception as e:
            return f"发生意外错误: {e}"

    # === WebUI ===
    with gr.Blocks() as demo:
        gr.Markdown(title)

        with gr.Row():
            record_start_btn = gr.Button("🎙️ 开始录音")
            record_stop_btn = gr.Button("⏹️ 停止录音并解析")
        result_output = gr.Textbox(label="语音输入", interactive=False)

        gr.Markdown("请选择您走迷宫的方式...")
        with gr.Row():
            turn_options = ["Left", "Right", "Straight"]
            cross1 = gr.Radio(label="第一个岔路口", choices=turn_options, value="Straight")
            cross2 = gr.Radio(label="第二个岔路口", choices=turn_options, value="Straight")
            cross3 = gr.Radio(label="第三个岔路口", choices=turn_options, value="Straight")

        answer_btn = gr.Button("解析文字模板")
        output_box = gr.Textbox(label="解析结果", interactive=False)

        gr.Markdown("选择起点和目标点")
        with gr.Row():
            start_point = gr.State(value="S")
            goal_points = gr.State(value=[])

            start_point_choices = ['S', 'A1_L', 'A1_S', 'A1_R', 'A2_LL', 'A2_LR', 'A2_SL', 'A2_SR', 'A2_RL', 'A2_RR', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10']
            goal_point_choices = start_point_choices.copy()

            start_point_dropdown = gr.Dropdown(label="选择起点", choices=start_point_choices, value="S")
            add_goal_point_dropdown = gr.Dropdown(label="添加目标点", choices=goal_point_choices)

            start_angle = gr.Slider(minimum=-180, maximum=180, step=1, value=0, label="选择起始角度")

            goal_points_list = gr.Textbox(label="目标点列表", interactive=False)

            def set_start_point(start): return start

            def add_goal_point(goal, goals):
                goals.append(goal)
                return goals, goals

            start_point_dropdown.change(set_start_point, inputs=[start_point_dropdown], outputs=[start_point])
            add_goal_point_dropdown.change(add_goal_point, inputs=[add_goal_point_dropdown, goal_points], outputs=[goal_points, goal_points_list])

        plan_btn = gr.Button("确定规划路线")
        plan_output = gr.Textbox(label="路径规划结果", interactive=False)

        record_start_btn.click(start_recording_and_update_button, inputs=[], outputs=[record_start_btn])
        record_stop_btn.click(stop_recording_and_update_button, inputs=[], outputs=[result_output, record_start_btn])
        answer_btn.click(get_turns_from_ui, inputs=[cross1, cross2, cross3], outputs=[output_box])
        plan_btn.click(plan_path_and_save, inputs=[start_point, goal_points, start_angle], outputs=[plan_output])

        with gr.Row():
            launch_btn = gr.Button("发车", visible=True)
            image = gr.Image("./frontend/car.png", label="我们发车啦~", visible=False)
        launch_btn.click(launch_ros, inputs=[], outputs=image)

        stop_btn = gr.Button("停车", visible=True, variant="stop")
        stop_btn.click(stop_car, inputs=[], outputs=[])

    demo.launch()

if __name__ == "__main__":
    main()
