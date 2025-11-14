import os
import torch
from ultralytics import YOLO

from sdks.novavision.src.base.download import Download
from sdks.novavision.src.base.logger import LoggerManager
from sdks.novavision.src.base.application import Application
from ultralytics.utils.torch_utils import select_device

logger = LoggerManager()
application = Application()

yolo11_weight = (
    "https://drive.google.com/file/d/105C582knZTmIrlXHbPPYgyYaVBtp_OoK/view?usp=sharing"
)

namedict = {
    0: "gauge",
    1: "center",
    2: "tip",
    3: "min",
    4: "max"
}

def download_weights(url, weight_name):
    if not os.path.exists(f"/storage/{weight_name}"):
        if Download.download_from_drive(url, f"/storage/{weight_name}") is None:
            logger.error(f"ReadGauge - Model ({weight_name}) download failed!")

    weight_path = f"/storage/{weight_name}"
    return weight_path


def load_model(config):
    weight_path = download_weights(url=yolo11_weight, weight_name="gauge-reader-medium-murphy.pt")
    config_device = application.get_param(config=config, name="ConfigDevice")
    use_half = application.get_param(config=config, name="Half")
    device = select_device(
        "cuda:0" if config_device == "GPU" and torch.cuda.is_available() else "cpu"
    )

    model = YOLO(weight_path).to(device)

    # Apply FP16 support if required.
    if use_half and device.type != "cpu":
        model.fuse()  # < yolo v8
        model = model.half()

    logger.info(f"Model loaded: {weight_path} | Device: {device} | FP16: {use_half}")
    return model