import re
import os
import pandas as pd
import torch
import cv2
import numpy as np
from pathlib import Path


class IemocapPreprocessor:
    def __init__(self, dataset_path: str, sessions_count: int = 5):
        self._dataset_path = Path(dataset_path)
        self._sessions_count = sessions_count
        self._info_line = re.compile(r"\[.+\]\n", re.IGNORECASE)

        self.face_detector = None
        self._setup_face_detector()

    def _setup_face_detector(self):
        try:
            prototxt_path = "deploy.prototxt"
            model_path = "res10_300x300_ssd_iter_140000.caffemodel"

            self.face_detector = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        except:
            print("Face detector not available. Install OpenCV DNN models.")
            self.face_detector = None

    def _extract_face_frame(self, video_path: Path, mid_time: float, target_size: tuple = (224, 224)):
        try:
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_idx = int(mid_time * fps)

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                return None

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.face_detector:
                h, w = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123])
                self.face_detector.setInput(blob)
                detections = self.face_detector.forward()

                if len(detections) > 0:
                    confidence = detections[0, 0, :, 2]
                    best_idx = np.argmax(confidence)
                    if confidence[best_idx] > 0.5:
                        box = detections[0, 0, best_idx, 3:7] * np.array([w, h, w, h])
                        x1, y1, x2, y2 = map(int, box)
                        face = frame[y1:y2, x1:x2]
                        if face.size > 0:
                            frame = cv2.resize(face, target_size)

            frame = cv2.resize(frame, target_size)
            frame = frame.astype(np.float32) / 255.0
            frame = torch.from_numpy(frame).permute(2, 0, 1)

            if frame.shape[0] == 1:
                frame = frame.repeat(3, 1, 1)

            return frame

        except Exception as e:
            print(f"Face extraction failed for {video_path}: {e}")
            return None

    def _parse_time_interval(self, info_line: str):
        parts = info_line.strip().split("\t")
        time_part = parts[0]
        wav_file_name = parts[1]
        emotion = parts[2]

        time_part = time_part.strip()[1:-1]
        start_str, end_str = [x.strip() for x in time_part.split("-")]
        start_time = float(start_str)
        end_time = float(end_str)

        return start_time, end_time, wav_file_name, emotion

    def generate_dataframe(self) -> pd.DataFrame:
        audios = []
        emotions = []
        texts = []
        faces = []
        dialog_ids = []
        video_files = []
        start_times = []
        end_times = []
        mid_times = []

        print("Processing IEMOCAP sessions...")

        for session in range(1, self._sessions_count + 1):
            session_dir = f"Session{session}"
            emo_eval_dir = self._dataset_path / session_dir / "dialog" / "EmoEvaluation"
            transcriptions_dir = self._dataset_path / session_dir / "dialog" / "transcriptions"
            videos_dir = self._dataset_path / session_dir / "sentences" / "wav"

            eval_files = [f for f in os.listdir(emo_eval_dir) if "Ses" in f]

            for file in eval_files:
                eval_path = emo_eval_dir / file
                transcription_path = transcriptions_dir / file

                if not os.path.exists(transcription_path):
                    continue

                with open(eval_path, "r") as eval_file, open(transcription_path, "r") as text_file:
                    eval_content = eval_file.read()
                    info_lines = re.findall(self._info_line, eval_content)

                    text_lines = sorted([
                        line for line in text_file.readlines()
                        if line.startswith("Ses") and line.split(" ")[0][-3:].isdigit()
                    ])

                    for info_line, text_line in zip(info_lines[1:], text_lines):
                        start_time, end_time, wav_file_name, emotion = self._parse_time_interval(info_line)

                        if emotion == "xxx":
                            continue

                        left_part, text = text_line.strip().split("]: ")
                        assert wav_file_name == left_part.split(" ")[0]

                        dialog_id = "_".join(wav_file_name.split("_")[:-1])
                        video_file = dialog_id + ".avi"
                        session_num = f"Session{session}"
                        video_path = (
                                self._dataset_path
                                / session_num
                                / "dialog"
                                / "avi"
                                / "DivX"
                                / video_file
                        )
                        mid_time = 0.5 * (start_time + end_time)

                        face_tensor = self._extract_face_frame(video_path, mid_time) if video_path.exists() else None

                        if face_tensor is None:
                            print(f"Skipping {wav_file_name}: no face detected")
                            continue

                        audios.append(wav_file_name)
                        emotions.append(emotion)
                        texts.append(text)
                        faces.append(face_tensor)
                        dialog_ids.append(dialog_id)
                        video_files.append(video_file)
                        start_times.append(start_time)
                        end_times.append(end_time)
                        mid_times.append(mid_time)

        df = pd.DataFrame({
            "audio": audios,
            "face": faces,
            "text": texts,
            "emotion": emotions,
            "dialog_id": dialog_ids,
            "video_file": video_files,
            "start_time": start_times,
            "end_time": end_times,
            "mid_time": mid_times,
        })

        print(f"Generated dataframe with {len(df)} valid face samples")
        print("Emotion distribution:")
        print(df['emotion'].value_counts())

        return df
