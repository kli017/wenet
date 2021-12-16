// Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "websocket/websocket_onnx_server.h"

#include <thread>
#include <utility>
#include <vector>

#include "boost/json/src.hpp"

#include "utils/log.h"

namespace wenet {

namespace beast = boost::beast;          // from <boost/beast.hpp>
namespace http = beast::http;            // from <boost/beast/http.hpp>
namespace websocket = beast::websocket;  // from <boost/beast/websocket.hpp>
namespace asio = boost::asio;            // from <boost/asio.hpp>
using tcp = boost::asio::ip::tcp;        // from <boost/asio/ip/tcp.hpp>
namespace json = boost::json;

OnnxConnectionHandler::OnnxConnectionHandler(
    tcp::socket&& socket, std::shared_ptr<FeaturePipelineConfig> feature_config,
    std::shared_ptr<DecodeOptions> decode_config,
    std::shared_ptr<OnnxDecodeResource> decode_resource)
    : ws_(std::move(socket)),
      feature_config_(std::move(feature_config)),
      decode_config_(std::move(decode_config)),
      decode_resource_(std::move(decode_resource)) {}

void OnnxConnectionHandler::OnSpeechStart() {
  LOG(INFO) << "Recieved speech start signal, start reading speech";
  got_start_tag_ = true;
  json::value rv = {{"status", "ok"}, {"type", "server_ready"}};
  ws_.text(true);
  ws_.write(asio::buffer(json::serialize(rv)));
  feature_pipeline_ = std::make_shared<FeaturePipeline>(*feature_config_);
  decoder_ = std::make_shared<OnnxAsrDecoder>(
      feature_pipeline_, decode_resource_, *decode_config_);
  // Start decoder thread
  decode_thread_ =
      std::make_shared<std::thread>(&OnnxConnectionHandler::DecodeThreadFunc, this);
}

void OnnxConnectionHandler::OnSpeechEnd() {
  LOG(INFO) << "Recieved speech end signal";
  CHECK(feature_pipeline_ != nullptr);
  feature_pipeline_->set_input_finished();
  got_end_tag_ = true;
}

void OnnxConnectionHandler::OnPartialResult(const std::string& result) {
  LOG(INFO) << "Partial result: " << result;
  json::value rv = {
      {"status", "ok"}, {"type", "partial_result"}, {"nbest", result}};
  ws_.text(true);
  ws_.write(asio::buffer(json::serialize(rv)));
}

void OnnxConnectionHandler::OnFinalResult(const std::string& result) {
  LOG(INFO) << "Final result: " << result;
  json::value rv = {
      {"status", "ok"}, {"type", "final_result"}, {"nbest", result}};
  ws_.text(true);
  ws_.write(asio::buffer(json::serialize(rv)));
}

void OnnxConnectionHandler::OnFinish() {
  // Send finish tag
  json::value rv = {{"status", "ok"}, {"type", "speech_end"}};
  ws_.text(true);
  ws_.write(asio::buffer(json::serialize(rv)));
}

void OnnxConnectionHandler::OnSpeechData(const beast::flat_buffer& buffer) {
  // Read binary PCM data
  int num_samples = buffer.size() / sizeof(int16_t);
  std::vector<float> pcm_data(num_samples);
  const int16_t* pdata = static_cast<const int16_t*>(buffer.data().data());
  for (int i = 0; i < num_samples; i++) {
    pcm_data[i] = static_cast<float>(*pdata);
    pdata++;
  }
  VLOG(2) << "Recieved " << num_samples << " samples";
  CHECK(feature_pipeline_ != nullptr);
  CHECK(decoder_ != nullptr);
  feature_pipeline_->AcceptWaveform(pcm_data);
}

std::string OnnxConnectionHandler::SerializeResult(bool finish) {
  json::array nbest;
  for (const DecodeResult& path : decoder_->result()) {
    json::object jpath({{"sentence", path.sentence}});
    if (finish) {
      json::array word_pieces;
      for (const WordPiece& word_piece : path.word_pieces) {
        json::object jword_piece({{"word", word_piece.word},
                                  {"start", word_piece.start},
                                  {"end", word_piece.end}});
        word_pieces.emplace_back(jword_piece);
      }
      jpath.emplace("word_pieces", word_pieces);
    }
    nbest.emplace_back(jpath);

    if (nbest.size() == nbest_) {
      break;
    }
  }
  return json::serialize(nbest);
}

void OnnxConnectionHandler::DecodeThreadFunc() {
  while (true) {
    DecodeState state = decoder_->Decode();
    if (state == DecodeState::kEndFeats) {
      decoder_->Rescoring();
      std::string result = SerializeResult(true);
      OnFinalResult(result);
      OnFinish();
      stop_recognition_ = true;
      break;
    } else if (state == DecodeState::kEndpoint) {
      decoder_->Rescoring();
      std::string result = SerializeResult(true);
      OnFinalResult(result);
      // If it's not continuous decoidng, continue to do next recognition
      // otherwise stop the recognition
      if (continuous_decoding_) {
        decoder_->ResetContinuousDecoding();
      } else {
        OnFinish();
        stop_recognition_ = true;
        break;
      }
    } else {
      if (decoder_->DecodedSomething()) {
        std::string result = SerializeResult(false);
        OnPartialResult(result);
      }
    }
  }
}

void OnnxConnectionHandler::OnError(const std::string& message) {
  json::value rv = {{"status", "failed"}, {"message", message}};
  ws_.text(true);
  ws_.write(asio::buffer(json::serialize(rv)));
  // Close websocket
  ws_.close(websocket::close_code::normal);
}

void OnnxConnectionHandler::OnText(const std::string& message) {
  json::value v = json::parse(message);
  if (v.is_object()) {
    json::object obj = v.get_object();
    if (obj.find("signal") != obj.end()) {
      json::string signal = obj["signal"].as_string();
      if (signal == "start") {
        if (obj.find("nbest") != obj.end()) {
          if (obj["nbest"].is_int64()) {
            nbest_ = obj["nbest"].as_int64();
          } else {
            OnError("integer is expected for nbest option");
          }
        }
        if (obj.find("continuous_decoding") != obj.end()) {
          if (obj["continuous_decoding"].is_bool()) {
            continuous_decoding_ = obj["continuous_decoding"].as_bool();
          } else {
            OnError(
                "boolean true or false is expected for "
                "continuous_decoding option");
          }
        }
        OnSpeechStart(); //初始化featurepipline decoder_
      } else if (signal == "end") {
        OnSpeechEnd();
      } else {
        OnError("Unexpected signal type");
      }
    } else {
      OnError("Wrong message header");
    }
  } else {
    OnError("Wrong protocol");
  }
}

void OnnxConnectionHandler::operator()() {
  try {
    // Accept the websocket handshake
    ws_.accept();
    for (;;) {
      // This buffer will hold the incoming message
      beast::flat_buffer buffer;
      // Read a message
      ws_.read(buffer);
      if (ws_.got_text()) {
        std::string message = beast::buffers_to_string(buffer.data());
        LOG(INFO) << message;
        OnText(message);
        if (got_end_tag_) {
          break;
        }
      } else {
        if (!got_start_tag_) {
          OnError("Start signal is expected before binary data");
        } else {
          if (stop_recognition_) {
            break;
          }
          OnSpeechData(buffer);
        }
      }
    }

    LOG(INFO) << "Read all pcm data, wait for decoding thread";
    if (decode_thread_ != nullptr) {
      decode_thread_->join();
    }
  } catch (beast::system_error const& se) {
    // This indicates that the session was closed
    if (se.code() != websocket::error::closed) {
      LOG(ERROR) << se.code().message();
    }
  } catch (std::exception const& e) {
    LOG(ERROR) << e.what();
  }
}

void WebSocketServerOnnx::Start() {
  try {
    auto const address = asio::ip::make_address("0.0.0.0");
    tcp::acceptor acceptor{ioc_, {address, static_cast<uint16_t>(port_)}};
    for (;;) {
      // This will receive the new connection
      tcp::socket socket{ioc_};
      // Block until we get a connection
      acceptor.accept(socket);
      // Launch the session, transferring ownership of the socket
      OnnxConnectionHandler handler(std::move(socket), feature_config_,
                                decode_config_, decode_resource_);
      std::thread t(std::move(handler));
      t.detach();
    }
  } catch (const std::exception& e) {
    LOG(FATAL) << e.what();
  }
}

}  // namespace wenet
