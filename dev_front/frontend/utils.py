import base64
from openai import OpenAI
import json


def llm_maze_infer(input_text, input_api_key):
    client = OpenAI(
        base_url="https://ai-gateway.vei.volces.com/v1",
        api_key = input_api_key
    )

    completion = client.chat.completions.create(
        model="doubao-1.5-vision-lite", # to use it, in web https://www.volcengine.com/docs/6893/1263410, 火山引擎大模型网关控制台，右上角编辑按钮，勾选即可
        messages=[
            {"role": "system", "content": "你现在来为我做一个数据处理，最终的处理结果应该和1-0 2-2 3-2差不多，其中第一个数字代表第几个岔路口，应该从一开始递增"},
            {"role": "user", "content": [
                {
                    "type": "text",
                    "text": "请将这段话解析为如下格式：数字-数字\n。第一个数字代表第几个岔路口，第二个数字代表左转或右转或直行，假设左转为0右转为1直行为2。{}".format(input_text)
                },
            ]}
        ],
    )
    return completion.choices[0].message.content

def llm_asr_infer(input_text, input_api_key):
    client = OpenAI(
        base_url="https://ai-gateway.vei.volces.com/v1",
        api_key = input_api_key
    )

    completion = client.chat.completions.create(
        model="doubao-1.5-vision-lite", # to use it, in web https://www.volcengine.com/docs/6893/1263410, 火山引擎大模型网关控制台，右上角编辑按钮，勾选即可
        messages=[
            {"role": "system", "content": "你现在是一个数据处理帮手，你现在来为我做一个数据映射，最终的处理结果应该和1-0 2-2 3-2差不多，后面的\n代表换行符。如果我说的是一号出口 请返回1-0/n2-0/n3-2 如果我说的是二号出口 请返回1-0/n2-0/n3-1 如果我说的是三号出口 请返回1-0/n2-1/n3-0 如果我说的是四号出口 请返回1-0/n2-1/n3-2 如果我说的是五号出口 请返回1-2/n2-0/n3-1"
            " 如果我说的是六号出口 请返回1-2/n2-1/n3-0 如果我说的是七号出口 请返回1-1/n2-0/n3-2 如果我说的是八号出口 请返回1-1/n2-0/n3-1 如果我说的是九号出口 请返回1-1/n2-1/n3-0 如果我说的是二号出口 请返回1-1/n2-1/n3-2"},
            {"role": "user", "content": [
                {
                    "type": "text",
                    "text": "{}".format(input_text)
                },
            ]}
        ],
    )
    return completion.choices[0].message.content

def llm_infer_validate(file_path):
    init_cross = 1
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                nums = line.strip().split('-')
                
                assert int(nums[0].strip()) == init_cross, f"Expected cross {init_cross} has instruction"
                assert int(nums[1].strip()) in {0, 1, 2}, f"Expected left 0 or right 1 or straight 2, got {nums[1].strip()}"
                
                init_cross += 1
        return True
    except AssertionError as e:
        print(f"Assertion failed: {e}")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False


if __name__ == "__main__":
    llm_maze_infer("你好，请你在第一个路口右转转，第二个路口左转。")