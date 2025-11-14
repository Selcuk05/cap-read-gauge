
from sdks.novavision.src.helper.package import PackageHelper
from components.ReadGauge.src.models.PackageModel import PackageModel, PackageConfigs, ConfigExecutor, ReadGaugeOutputs, ReadGaugeResponse, ReadGauge, OutputDetections, OutputValue


def build_response(context):
    output_detections = OutputDetections(value=context.detections)
    output_value = OutputValue(value=context.projected_value)
    _outputs = ReadGaugeOutputs(outputValue=output_value, outputDetections=output_detections)
    packageResponse = ReadGaugeResponse(outputs=_outputs)
    packageExecutor = ReadGauge(value=packageResponse)
    executor = ConfigExecutor(value=packageExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel