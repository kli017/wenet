#include "tcp/tcp_server.h"

#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>
#include <thread>
#include <utility>
#include <vector>

#include "utils/log.h"

namespace wenet {

void TcpServer::Start() {
    try {
        Bind();
        LOG(INFO) << "Waiting for client connect...";
        while (true) {
            int32_t fd = Accept();
            std::thread t(&TcpServer::HandleThreadFunc, this, fd);
            t.detach();
        }
    } catch (const std::exception& e) {
        LOG(INFO) << e.what();
    }
}

void TcpServer::Bind() {
    addr_.sin_addr.s_addr = INADDR_ANY;
    addr_.sin_port = htons(port_);
    addr_.sin_family = AF_INET;

    socket_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_fd_ == -1) LOG(FATAL) << "Cannot create TCP socket!";

    int32_t flag = 1;
    int32_t len = sizeof(int32_t);

    if (setsockopt(socket_fd_, SOL_SOCKET, SO_REUSEADDR, &flag, len) == -1)
        LOG(FATAL) << "Cannot set socket options!";
    if (bind(socket_fd_, (struct sockaddr*)&addr_, sizeof(addr_)) == -1)
        LOG(FATAL) << "Cannot bind to port " << port_
                << "(Please check if has been token)";
    if (listen(socket_fd_, 1) == -1) LOG(FATAL) << "Cannot listen port";

    LOG(INFO) << "Server: Listing on port: " << port_;
}

int32_t TcpServer::Accept() {
    socklen_t len;
    len = sizeof(struct sockaddr);
    int32_t fd = accept(socket_fd_, (struct sockaddr*)&addr_, &len);
    struct sockaddr_storage addr;
    char ipstr[20];

    len = sizeof(addr);
    getpeername(fd, (struct sockaddr*)&addr, &len);
    struct sockaddr_in* s = (struct sockaddr_in*)&addr;
    inet_ntop(AF_INET, &s->sin_addr, ipstr, sizeof ipstr);
    LOG(INFO) << "Accepted connection from: " << ipstr;
    return fd;
}

void TcpServer::HandleThreadFunc(int32_t fd) {
  // Handle thread function
  TcpConnectionHandler handler(std::move(fd), feature_config_, decode_config_,
                               decode_resource_);
  handler();
}

class TcpConnectionHandler::TcpConnection {
public:
    explicit TcpConnection(int32_t socket_fd)
        : connected_(true), socket_fd_(socket_fd) {
        InitAddress();
    }

    void Read(std::vector<char>& buf) { Read(buf.data(), buf.size()); }

    bool Write(const std::string& message) {
        const char* p = message.c_str();
        int32_t to_write = message.size();
        int32_t wrote = 0;

        while (to_write > 0) {
            int32_t ret =
                write(socket_fd_, static_cast<const void*>(p + wrote), to_write);
            if (ret <= 0) return false;
            to_write -= ret;
            wrote += ret;
        }
        return true;
    }

  bool IsConnected() { return connected_; }

  void Disconnect() {
      connected_ = false;
      if (socket_fd_ != -1) close(socket_fd_);
      socket_fd_ = -1;
  }

  std::string& GetAddress() { return address_; }

private:
    void InitAddress() {
        int32_t flag = 1;
        setsockopt(socket_fd_, IPPROTO_TCP, TCP_NODELAY, (void*)&flag,
                  sizeof(int32_t));
        struct sockaddr_storage addr;
        char ipstr[20];

        socklen_t len = sizeof(addr);
        getpeername(socket_fd_, (struct sockaddr*)&addr, &len);

        struct sockaddr_in* s = (struct sockaddr_in*)&addr;
        inet_ntop(AF_INET, &s->sin_addr, ipstr, sizeof ipstr);
        address_.assign(ipstr);
    }

    void Read(char* data, const int32_t size) {
        int32_t to_read = size;
        int32_t readed = 0;

        while (to_read > 0) {
            int32_t ret =
                read(socket_fd_, static_cast<void*>(data + readed), to_read);
            if (ret <= 0) {
              // if ret <= 0, we thick the read buffer is shutdown.
              connected_ = false;
              break;
            }
            to_read -= ret;
            readed += ret;
        }
    }

    void ReadHead(std::string& id) {
        char data[35];
        Read(data, 2);
        if (isspace(data[0]) && isspace(data[1])) {
            id.assign("");
            return;
        }

        if (!IsConnected()) return;
        Read(data + 2, 32);
        if (!IsConnected()) return;

        id.clear();
        id.assign(data);
        id.assign(id.substr(0, 32));
    }
    
    bool connected_ = false;
    int32_t socket_fd_ = -1;
    std::string address_ = "";

};  // Class TcpConnection

  TcpConnectionHandler::TcpConnectionHandler(
      int32_t socket_fd, std::shared_ptr<FeaturePipelineConfig> feature_config,
      std::shared_ptr<DecodeOptions> decode_config,
      std::shared_ptr<DecodeResource> decode_resource)
      : connection_(std::make_shared<TcpConnection>(socket_fd)),
        feature_config_(std::move(feature_config)),
        decode_config_(std::move(decode_config)),
        decode_resource_(std::move(decode_resource)) {}

  void TcpConnectionHandler::OnSpeechStart() {
      LOG(INFO) << "Recieved speech start signal, start reading speech";
      feature_pipeline_ = std::make_shared<FeaturePipeline>(*feature_config_);
      decoder_ = std::make_shared<TorchAsrDecoder>(
          feature_pipeline_, decode_resource_, *decode_config_);
      // Start decoder thread
      decode_thread_ = std::make_shared<std::thread>(
          &TcpConnectionHandler::DecodeThreadFunc, this);
  }

  void TcpConnectionHandler::OnSpeechEnd() {
      LOG(INFO) << "Recieved speech end signal";
      CHECK(feature_pipeline_ != nullptr);
      feature_pipeline_->set_input_finished();
      decode_thread_->join();
      connection_->Disconnect();
      LOG(WARNING) << connection_->GetAddress() << " disconnected.";
  }

  void TcpConnectionHandler::OnSpeechData(std::vector<char>& buf) {
      // Read binary PCM data
      int num_samples = buf.size() / sizeof(int16_t);
      std::vector<float> pcm_data(num_samples);
      const int16_t* pdata = reinterpret_cast<const int16_t*>(buf.data());
      for(int i = 0; i < num_samples; i++){
          pcm_data[i] = static_cast<float>(*pdata);
          pdata++;
      }
      VLOG(2)<<"Recived"<<num_samples<<"samples";
      CHECK(feature_pipeline_!=nullptr);
      CHECK(decoder_!=nullptr);
      feature_pipeline_->AcceptWaveform(pcm_data);
  }

  void TcpConnectionHandler::OnPartialResult(const std::string& result) {
      LOG(INFO) << "Partial result: " << result;
      if (connection_->Write(result)) connection_->Write("\r");
  }

  void TcpConnectionHandler::OnFinalResult(const std::string& result) {
      LOG(INFO) << "Final result: " << result;
      if (connection_->Write(result)) connection_->Write("\n");
  }


// std::string ConnectionHandler::SerializeResult(bool finish) {
//   json::array nbest;
//   for (const DecodeResult& path : decoder_->result()) {
//     json::object jpath({{"sentence", path.sentence}});
//     if (finish) {
//       json::array word_pieces;
//       for (const WordPiece& word_piece : path.word_pieces) {
//         json::object jword_piece({{"word", word_piece.word},
//                                   {"start", word_piece.start},
//                                   {"end", word_piece.end}});
//         word_pieces.emplace_back(jword_piece);
//       }
//       jpath.emplace("word_pieces", word_pieces);
//     }
//     nbest.emplace_back(jpath);

//     if (nbest.size() == nbest_) {
//       break;
//     }
//   }
//   return json::serialize(nbest);
// }

void TcpConnectionHandler::DecodeThreadFunc() {
    std::string last_result;
    std::string result;
    while (connection_->IsConnected()){
        DecodeState state = decoder_->Decode();
        if(state==DecodeState::kEndFeats){
            decoder_->Rescoring();
            std::string result = decoder_->result()[0].sentence;
            OnFinalResult(result);

        }else if(state==DecodeState::kEndpoint){
            decoder_->Rescoring();
            std::string result = decoder_->result()[0].sentence;
            OnFinalResult(result);

        }else{
            if(decoder_->DecodedSomething()){
                std::string result = decoder_->result()[0].sentence;
                OnPartialResult(result);
            }
    }
    
}

void TcpConnectionHandler::operator()() {
    OnSpeechStart();

    std::vector<char> buf(320);
    try{
        while(connection_->IsConnected()){
            connection_->Read(buf);
            OnSpeechData(buf);
            buf.clear();
            buf.resize(320);
        }
        OnSpeechEnd();
    }catch(std::exception& e){
        LOG(WARNING) << e.what();
    }
}

}  // namespace wenet