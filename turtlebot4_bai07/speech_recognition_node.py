"""Phần 3: speech -> text with Vosk (offline, Vietnamese); the text goes out on /voice/text.

    ros2 run turtlebot4_bai07 speech_recognition_node                              # microphone
    ros2 run turtlebot4_bai07 speech_recognition_node --ros-args -p source:=none   # only /voice/wav
    ros2 topic pub --once /voice/wav std_msgs/msg/String "{data: /path/command.wav}"  # 16-bit mono wav

Model: vosk-model-small-vn-0.4 (32 MB), downloaded to ~/.cache/turtlebot4_bai07 on first run.
The microphone is read with sounddevice at 16 kHz; Vosk ends an utterance at the pause after it and
the final text is published once. With grammar:=true (default) the recognizer only chooses among
the command sentences built from regions.yaml (command words, region names and letters, and a few
places that are not on the map, so that those are heard as such and reported, not mistaken for a
region); grammar:=false recognizes free Vietnamese speech.
"""
import json
import os
import queue
import threading
import urllib.request
import wave
import zipfile

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from std_msgs.msg import String

MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-vn-0.4.zip"
CACHE = os.path.expanduser("~/.cache/turtlebot4_bai07")

# Letters as they are read in Vietnamese (words of the Vosk vocabulary). Only the letters of the
# regions go into the grammar: one-syllable letters are easily confused with each other.
LETTER_WORDS = {"A": ["a"], "B": ["bê", "bờ"], "C": ["xê", "cờ"], "D": ["dê", "đê"], "E": ["e"],
                "F": ["ép"], "G": ["gờ"], "H": ["hát"], "I": ["i"], "J": ["gi"], "K": ["ca"], "L": ["lờ"],
                "M": ["mờ"], "N": ["en"], "O": ["o"], "P": ["pê"], "Q": ["quy"], "R": ["rờ"], "T": ["tê"],
                "U": ["u"], "V": ["vê"], "X": ["ích"], "Y": ["y"]}
VERBS = ["đi đến", "đi tới", "di chuyển đến", "di chuyển tới", "đến", "tới", "về", "hãy đi đến", "robot đi đến"]
FROM = ["từ", "đi từ", "di chuyển từ"]
STOP = ["dừng lại", "dừng", "hủy", "robot dừng lại"]
# Places that are not on the map: in the vocabulary so they are heard and reported as unknown.
OTHER_PLACES = ["phòng ngủ", "phòng ăn", "nhà bếp", "nhà vệ sinh", "phòng khách", "sân"]


def build_grammar(regions_file, extra):
    with open(regions_file) as f:
        regions = yaml.safe_load(f)["regions"]
    places = list(OTHER_PLACES) + list(extra)
    for r in regions:
        places += [r["name"].lower()] + [a.lower() for a in r.get("aliases", [])]
    places += [f"khu vực {w}" for r in regions for w in LETTER_WORDS.get(str(r["label"]).upper(), [])]
    places = list(dict.fromkeys(places))
    phrases = [f"{v} {p}" for v in VERBS for p in places] + places + STOP
    phrases += [f"{f} {a} {t} {b}" for f in FROM for t in ("đến", "tới") for a in places for b in places if a != b]
    return phrases + ["[unk]"]


def ensure_model(path):
    if os.path.isdir(path):
        return path
    os.makedirs(CACHE, exist_ok=True)
    zip_path = os.path.join(CACHE, os.path.basename(MODEL_URL))
    print(f"downloading {MODEL_URL} (32 MB) ...", flush=True)
    urllib.request.urlretrieve(MODEL_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(CACHE)
    os.remove(zip_path)
    return path


class SpeechRecognitionNode(Node):
    def __init__(self):
        super().__init__("speech_recognition_node")
        share = get_package_share_directory("turtlebot4_bai07")
        self.declare_parameter("model", os.path.join(CACHE, "vosk-model-small-vn-0.4"))
        self.declare_parameter("source", "mic")         # mic | none (only /voice/wav)
        self.declare_parameter("device", "")            # sounddevice input (index or name), "" = default
        self.declare_parameter("sample_rate", 16000)
        self.declare_parameter("grammar", True)
        self.declare_parameter("regions_file", os.path.join(share, "maps", "regions.yaml"))
        self.declare_parameter("extra_places", [""])
        p = lambda name: self.get_parameter(name).value  # noqa: E731

        from vosk import KaldiRecognizer, Model, SetLogLevel
        SetLogLevel(-1)
        self.KaldiRecognizer = KaldiRecognizer
        self.model = Model(ensure_model(p("model")))
        self.grammar = None
        if p("grammar"):
            phrases = build_grammar(p("regions_file"), [e for e in p("extra_places") if e])
            self.grammar = json.dumps(phrases, ensure_ascii=False)
            self.get_logger().info(f"grammar: {len(phrases)} command sentences from {p('regions_file')}")

        self.text_pub = self.create_publisher(String, "voice/text", 10)
        self.partial_pub = self.create_publisher(String, "voice/partial", 10)
        self.create_subscription(String, "voice/wav", lambda m: self.jobs.put(("wav", m.data)), 10)
        self.jobs = queue.Queue()
        self.running = True
        self.stream = None
        self.rate = int(p("sample_rate"))
        if p("source") == "mic":
            self._open_mic(p("device"))
        self.worker = threading.Thread(target=self._work, daemon=True)
        self.worker.start()

    def recognizer(self, rate):
        if self.grammar:
            return self.KaldiRecognizer(self.model, rate, self.grammar)
        return self.KaldiRecognizer(self.model, rate)

    def _open_mic(self, device):
        import sounddevice as sd
        dev = int(device) if str(device).isdigit() else (device or None)
        info = sd.query_devices(dev, "input")
        try:
            sd.check_input_settings(device=dev, samplerate=self.rate, channels=1, dtype="int16")
        except Exception:  # noqa: BLE001 - the device does not do 16 kHz: use its own rate
            self.rate = int(info["default_samplerate"])
        self.stream = sd.RawInputStream(device=dev, samplerate=self.rate, channels=1, dtype="int16",
                                        blocksize=self.rate // 5, callback=self._on_audio)
        self.stream.start()
        self.get_logger().info(f"listening on '{info['name']}' at {self.rate} Hz - speak a command")

    def _on_audio(self, data, frames, time_info, status):
        self.jobs.put(("mic", bytes(data)))

    def _work(self):
        mic = self.recognizer(self.rate)
        last_partial = ""
        while self.running:
            try:
                kind, data = self.jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            if kind == "mic":
                if mic.AcceptWaveform(data):
                    self._publish(json.loads(mic.Result())["text"], "mic")
                    last_partial = ""
                else:
                    partial = json.loads(mic.PartialResult())["partial"]
                    if partial and partial != last_partial:
                        last_partial = partial
                        self.partial_pub.publish(String(data=partial))
            elif kind == "wav":
                self._publish(self._recognize_file(data), os.path.basename(data))

    def _recognize_file(self, path):
        try:
            with wave.open(path) as wf:
                if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
                    self.get_logger().error(f"{path}: need 16-bit mono wav")
                    return ""
                rec = self.recognizer(wf.getframerate())
                while True:
                    chunk = wf.readframes(4000)
                    if not chunk:
                        break
                    rec.AcceptWaveform(chunk)
                return json.loads(rec.FinalResult())["text"]
        except (OSError, wave.Error) as e:
            self.get_logger().error(f"{path}: {e}")
            return ""

    def _publish(self, text, source):
        text = " ".join(w for w in text.split() if w != "[unk]")
        if not text:
            return
        self.get_logger().info(f"heard ({source}): \"{text}\"")
        self.text_pub.publish(String(data=text))

    def destroy_node(self):
        self.running = False
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = SpeechRecognitionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
