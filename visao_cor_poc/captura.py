import cv2
import numpy as np
import tempfile
import os
from typing import Tuple, Optional, List

class CapturadorWebcam:
    """POC: Captura de frames via OpenCV VideoCapture @ 1 fps"""

    def __init__(self, camera_id: int = 0, fps: int = 1):
        self.cap = cv2.VideoCapture(camera_id)
        self.fps = fps
        self.frame_delay = int(1000 / fps)
        self.set_auto_exposure()

    def set_auto_exposure(self):
        """Auto-exposure travado (POC)"""
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -8)

    def capturar_frame(self) -> Optional[np.ndarray]:
        """Captura frame com delay configurado"""
        ret, frame = self.cap.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

    def liberar(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.liberar()


class CapturadorArquivo:
    """POC: Captura de frames a partir de foto ou vídeo enviado pelo usuário"""

    @staticmethod
    def de_foto(file_bytes: bytes) -> Optional[np.ndarray]:
        """Decodifica imagem a partir de bytes, retorna RGB"""
        arr = np.frombuffer(file_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    @staticmethod
    def de_video(file_bytes: bytes, ext: str = '.mp4') -> List[np.ndarray]:
        """Extrai todos os frames de um vídeo a partir de bytes, retorna lista RGB"""
        tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        try:
            tmp.write(file_bytes)
            tmp.flush()
            tmp.close()

            cap = cv2.VideoCapture(tmp.name)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
        finally:
            os.unlink(tmp.name)

        return frames


if __name__ == '__main__':
    with CapturadorWebcam(fps=1) as capturador:
        for _ in range(5):
            frame = capturador.capturar_frame()
            if frame is not None:
                print(f"Frame capturado: {frame.shape}")
