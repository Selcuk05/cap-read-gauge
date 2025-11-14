from pydantic import Field, validator
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import (
    Package,
    Image,
    Inputs,
    Configs,
    Outputs,
    Response,
    Request,
    Output,
    Input,
    Config,
    Detection,
)


class InputImage(Input):
    name: Literal["inputImage"] = "inputImage"
    value: Union[List[Image], Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get('value')
        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"

    class Config:
        title = "Image"


class OutputReading(Output):
    name: Literal["outputReading"] = "outputReading"
    value: dict
    type: str = "object"

    class Config:
        title = "Reading"


class OutputDetections(Output):
    name: Literal["outputDetections"] = "outputDetections"
    value: List[Detection]
    type: str = "object"

    class Config:
        title = "Detections"






class ConfigGaugeMinimumValue(Config):
    name: Literal["GaugeMinimumValue"] = "GaugeMinimumValue"
    value: float = Field(ge=0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Minimum Value"


class ConfigGaugeMaximumValue(Config):
    name: Literal["GaugeMaximumValue"] = "GaugeMaximumValue"
    value: float = Field(ge=0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Maximum Value"


class ConfigUseLongerArcTrue(Config):
    name: Literal["UseLongerArcTrue"] = "UseLongerArcTrue"
    value: Literal[True] = True
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Yes"


class ConfigUseLongerArcFalse(Config):
    name: Literal["UseLongerArcFalse"] = "UseLongerArcFalse"
    value: Literal[False] = False
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "No"


class ConfigUseLongerArc(Config):
    name: Literal["UseLongerArc"] = "UseLongerArc"
    value: Union[ConfigUseLongerArcTrue, ConfigUseLongerArcFalse]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Use Longer Arc"

class ConfigReadingPrecision(Config):
    name: Literal["ReadingPrecision"] = "ReadingPrecision"
    value: int = Field(ge=0, le=12, default=2)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Reading Precision"






class ConfigConfidenceThreshold(Config):
    """
    (0.0-1.0) Represents the confidence threshold value.
    """

    name: Literal["ConfidenceThreshold"] = "ConfidenceThreshold"
    value: float = Field(default=0.3, ge=0, le=1)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Confidence Threshold"


class ConfigIOUThreshold(Config):
    """
    (0.0-1.0) Represents the overlap threshold value.
    """

    name: Literal["IOUThreshold"] = "IOUThreshold"
    value: float = Field(default=0.3, ge=0, le=1)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "IOU Threshold"


class ConfigHalfTrue(Config):
    name: Literal["HalfTrue"] = "HalfTrue"
    value: Literal[True] = True
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Enable"


class ConfigHalfFalse(Config):
    name: Literal["HalfFalse"] = "HalfFalse"
    value: Literal[False] = False
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Disable"


class ConfigHalf(Config):
    """
    It enables half-precision (FP16) inference, which can speed up model inference.
    """

    name: Literal["Half"] = "Half"
    value: Union[ConfigHalfTrue, ConfigHalfFalse]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "Half"


class ConfigDeviceGPU(Config):
    name: Literal["ConfigDeviceGPU"] = "ConfigDeviceGPU"
    configHalf: ConfigHalf
    value: Literal["GPU"] = "GPU"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "GPU"


class ConfigDeviceCPU(Config):
    name: Literal["ConfigDeviceCPU"] = "ConfigDeviceCPU"
    value: Literal["CPU"] = "CPU"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "CPU"


class ConfigDevice(Config):
    """
    It refers to whether the model should run on a CPU or a GPU.
    You can select the device type for inference or training process.
    """

    name: Literal["ConfigDevice"] = "ConfigDevice"
    value: Union[ConfigDeviceCPU, ConfigDeviceGPU]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "Device"


class ReadGaugeInputs(Inputs):
    inputImage: InputImage


class ReadGaugeOutputs(Outputs):
    outputReading: OutputReading
    outputDetections: OutputDetections


class ReadGaugeConfigs(Configs):
    configDevice: ConfigDevice
    configConfidenceThreshold: ConfigConfidenceThreshold
    configIOUThreshold: ConfigIOUThreshold

    gaugeMinimumValue: ConfigGaugeMinimumValue
    gaugeMaximumValue: ConfigGaugeMaximumValue
    useLongerArc: ConfigUseLongerArc
    readingPrecision: ConfigReadingPrecision

    class Config:
        title = "Read Gauge Configurations"


class ReadGaugeRequest(Request):
    inputs: Optional[ReadGaugeInputs]
    configs: ReadGaugeConfigs

    class Config:
        json_schema_extra = {"target": "configs"}


class ReadGaugeResponse(Response):
    outputs: ReadGaugeOutputs


class ReadGauge(Config):
    name: Literal["ReadGauge"] = "ReadGauge"
    value: Union[ReadGaugeRequest, ReadGaugeResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Read Gauge"
        json_schema_extra = {"target": {"value": 0}}


class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[ReadGauge]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Task"
        json_schema_extra = {"target": "value"}


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["capsule"] = "capsule"
    name: Literal["ReadGauge"] = "ReadGauge"
