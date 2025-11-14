
from sdks.novavision.src.helper.package import PackageHelper
from components.ReadGauge.src.models.PackageModel import PackageModel, PackageConfigs, ConfigExecutor, ReadGaugeOutputs, ReadGaugeResponse, ReadGauge, OutputDetections, OutputReading


def build_response(context):
    output_detections = OutputDetections(value=context.detections)
    output_reading = OutputReading(value=context.reading)
    _outputs = ReadGaugeOutputs(outputReading=output_reading, outputDetections=output_detections)
    packageResponse = ReadGaugeResponse(outputs=_outputs)
    packageExecutor = ReadGauge(value=packageResponse)
    executor = ConfigExecutor(value=packageExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel