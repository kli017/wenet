#include "decoder/params.h"
#include "utils/log.h"
#include "websocket/websocket_onnx_server.h"

DEFINE_int32(port, 10086, "websocket listening port");

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  auto decode_config = wenet::InitDecodeOptionsFromFlags();
  auto feature_config = wenet::InitFeaturePipelineConfigFromFlags();
  auto decode_resource = wenet::InitOnnxDecodeResourceFromFlags();

  wenet::WebSocketServer server(FLAGS_port, feature_config, decode_config,
                                decode_resource);
  LOG(INFO) << "Listening at port " << FLAGS_port;
  server.Start();
  return 0;
}