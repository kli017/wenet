// Author lk9171@gmail.com

#ifndef DECODER_ONNX_ASR_MODEL_H_
#define DECODER_ONNX_ASR_MODEL_H_

#include <memory>
#include <string>

#include "utils/utils.h"
#include "torch/script.h"
#include "torch/torch.h"
#include "third-party/onnx/include/session/onnxruntime_cxx_api.h"
#include "third-party/onnx/include/session/onnxruntime_run_options_config_keys.h"

namespace wenet {

class OnnxAsrModel {
public:
    using OnnxModule = Ort::Session;

    OnnxAsrModel() =  default;

    void Read( const std::string& torch_model_path,
        const std::string& encoder_model_path,
        const std::string& ctc_model_path,
        const std::string& decoder_model_path,
        int num_threads = 1);
    int right_context() const { return right_context_; }
    int subsampling_rate() const { return subsampling_rate_; }
    int sos() const { return sos_; }
    int eos() const { return eos_; }
    bool is_bidirectional_decoder() const { return is_bidirectional_decoder_; }
    void init_params();

    std::shared_ptr<OnnxModule> encoder_session() const { return encoder_module_; }
    std::shared_ptr<OnnxModule> ctc_session() const { return ctc_module_; }
    std::shared_ptr<OnnxModule> decoder_session() const { return decoder_module_; }

private:
    std::shared_ptr<OnnxModule> encoder_module_ = nullptr;
    std::shared_ptr<OnnxModule> ctc_module_ = nullptr;
    std::shared_ptr<OnnxModule> decoder_module_ = nullptr;
    std::shared_ptr<>
    int right_context_ = 1;
    int subsampling_rate_ = 1;
    int sos_ = 0;
    int eos_ = 0;
    bool is_bidirectional_decoder_ = false;

public:
    WENET_DISALLOW_COPY_AND_ASSIGN(OnnxAsrModel);
}   

} //namespace

#endif