// TCP_SERVER

#include "decoder/params.h"
#include "tcp/tcp_server.h"
#include "utils/log.h"

DEFINE_int32(port, 10010, "websocket listening port");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  auto decode_config = wenet::InitDecodeOptionsFromFlags();
  auto feature_config = wenet::InitFeaturePipelineConfigFromFlags();
  auto decode_resource = wenet::InitDecodeResourceFromFlags();


  wenet::TcpServer server(FLAGS_port, feature_config, decode_config, 
                        decode_resource);
  LOG(INFO) << "Listening at port " << FLAGS_port;
  server.Start();
  return 0;
}