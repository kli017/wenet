#ifndef TCP_TCP_SERVER_H_
#define TCP_TCP_SERVER_H_

#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include "decoder/torch_asr_decoder.h"
#include "decoder/torch_asr_model.h"
#include "frontend/feature_pipeline.h"
#include "utils/log.h"

namespace wenet {

class TcpConnectionHandler {
 public:
  TcpConnectionHandler(int32_t socket_fd,
                       std::shared_ptr<FeaturePipelineConfig> feature_config,
                       std::shared_ptr<DecodeOptions> decode_config,
                       std::shared_ptr<DecodeResource> decode_resource);
  void operator()();

 private:
  void OnSpeechStart();
  void OnSpeechEnd();
  void OnSpeechData(std::vector<char>& buf);
  void OnError(const std::string& message);
  void OnPartialResult(const std::string& result);
  void OnFinalResult(const std::string& result);
  void DecodeThreadFunc();


  std::shared_ptr<FeaturePipelineConfig> feature_config_;
  std::shared_ptr<DecodeOptions> decode_config_;
  std::shared_ptr<DecodeResource> decode_resource_;

  class TcpConnection;

  std::shared_ptr<TcpConnection> connection_ = nullptr;
  std::shared_ptr<FeaturePipeline> feature_pipeline_ = nullptr;
  std::shared_ptr<TorchAsrDecoder> decoder_ = nullptr;
  std::shared_ptr<std::thread> decode_thread_ = nullptr;

  
};

class TcpServer {
 public:
  TcpServer(int32_t port, std::shared_ptr<FeaturePipelineConfig> feature_config,
            std::shared_ptr<DecodeOptions> decode_config,
            std::shared_ptr<DecodeResource> decode_resource)
      : port_(port),
        feature_config_(std::move(feature_config)),
        decode_config_(std::move(decode_config)),
        decode_resource_(std::move(decode_resource)){};

  void Start();

 private:
  void Bind();
  int32_t Accept();
  void HandleThreadFunc(int32_t fd);  // tcp accept 产生fd

  int32_t port_;               // 端口号（打印）
  int32_t socket_fd_;          // sokcet file description
  struct ::sockaddr_in addr_;  // Ip + 端口 + tcp协议
  std::shared_ptr<FeaturePipelineConfig> feature_config_;
  std::shared_ptr<DecodeOptions> decode_config_;
  std::shared_ptr<DecodeResource> decode_resource_;

  WENET_DISALLOW_COPY_AND_ASSIGN(TcpServer);  // 宏防止隐式初始化？
};

}  // namespace wenet

#endif  // TCP_TCP_SERVER_H_
