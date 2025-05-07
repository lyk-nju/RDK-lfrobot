import hashlib, hmac, base64, json, time, threading
from websocket import create_connection, WebSocketConnectionClosedException
from urllib.parse import quote
import pyaudio
import wave

class ASRClient:
    def __init__(self, appid, api_key, url):
        self.APPID = appid
        self.API_KEY = api_key
        self.URL = url

        self.ws = None
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.recording = False
        self.full_sentence = ""
        self.last_text = ""
        self.punctuation = ""
        self.asr_result = ""

        self.CHUNK = 1280
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000

        self._init_ws()

    def _init_ws(self):
        ts = str(int(time.time()))
        base_str = (self.APPID + ts).encode("utf-8")
        base_md5 = hashlib.md5(base_str).hexdigest().encode("utf-8")
        signa = hmac.new(self.API_KEY.encode("utf-8"), base_md5, hashlib.sha1).digest()
        signa = base64.b64encode(signa).decode("utf-8")
        url = f"{self.URL}?appid={self.APPID}&ts={ts}&signa={quote(signa)}"
        self.ws = create_connection(url)
        threading.Thread(target=self._recv, daemon=True).start()

    def _recv(self):
        try:
            while self.ws.connected:
                result = self.ws.recv()
                if not result:
                    break
                result_dict = json.loads(result)
                if result_dict["action"] == "result":
                    try:
                        data_json = json.loads(result_dict["data"])
                        words = [
                            ws["cw"][0]["w"]
                            for rt in data_json.get("cn", {}).get("st", {}).get("rt", [])
                            for ws in rt.get("ws", [])
                            if ws.get("cw")
                        ]
                        final_text = ''.join(words)
                        self.last_text = self.punctuation
                        self.punctuation = final_text
                    except Exception as e:
                        print("解析识别结果出错:", e)
                elif result_dict["action"] == "error":
                    print("识别错误:", result_dict)
                    break
        except WebSocketConnectionClosedException:
            print("连接已关闭")
        finally:
            self.asr_result = self.last_text + self.punctuation
            print("完整识别结果：", self.asr_result)

    def _record_audio(self):
        self.stream = self.p.open(format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  input=True,
                                  frames_per_buffer=self.CHUNK)
        print("开始录音")
        try:
            while self.recording:
                data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                self.ws.send(data)
                time.sleep(0.04)
        finally:
            if self.ws.connected:
                self.ws.send(json.dumps({"end": True}))
            if self.stream.is_active():
                self.stream.stop_stream()
            self.stream.close()

    def start_recording(self):
        self.recording = True
        self.audio_data = []
        threading.Thread(target=self._record_audio, daemon=True).start()

    def stop_recording(self):
        self.recording = False

    def get_result(self):
        # Save audio data to a temporary file
        temp_file = "temp_audio.wav"
        wf = wave.open(temp_file, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self.audio_data))
        wf.close()
        return self.asr_result, temp_file

    def close(self):
        self.stop_recording()
        if self.ws:
            self.ws.close()
        self.p.terminate()


if __name__ == '__main__':
    client = ASRClient(appid="9fe217f8", api_key="0b6a94e0344907568eca1f09d9828c2c", url="ws://rtasr.xfyun.cn/v1/ws")
    client.start_recording()
    time.sleep(5)  # 录音5秒
    client.stop_recording()
    time.sleep(2)  # 等待识别结果
    print("识别结果：", client.get_result())
    client.close()
