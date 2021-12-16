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
    using TorchModule = torch::jit::script::Module;

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


    std::shared_ptr<OnnxModule> encoder_session() const { return encoder_module_; }
    std::shared_ptr<OnnxModule> ctc_session() const { return ctc_module_; }
    std::shared_ptr<OnnxModule> decoder_session() const { return decoder_module_; }

    std::vector<const char*> encoder_input_node_names() const {return encoder_input_node_names_;}
    std::vector<const char*> encoder_output_node_names() const {return encoder_output_node_names_;}
    
    std::vector<const char*> decoder_input_node_names() const {return decoder_input_node_names_;}
    std::vector<const char*> decoder_output_node_names() const {return decoder_output_node_names_;}
    
    std::vector<const char*> ctc_input_node_names() const {return ctc_input_node_names_;}
    std::vector<const char*> ctc_output_node_names() const {return ctc_output_node_names_;}

private:
    void get_dims_and_names(Ort::Session* session, 
                          std::vector<std::vector<int64_t>>& input_node_dims, 
                          std::vector<const char*>& input_node_names,
                          std::vector<const char*>& output_node_names);
    void init_dims_and_names();

    std::shared_ptr<OnnxModule> encoder_module_ = nullptr;
    std::shared_ptr<OnnxModule> ctc_module_ = nullptr;
    std::shared_ptr<OnnxModule> decoder_module_ = nullptr;
    std::shared_ptr<TorchModule> module_ = nullptr;

    int right_context_ = 1;
    int subsampling_rate_ = 1;
    int sos_ = 0;
    int eos_ = 0;
    bool is_bidirectional_decoder_ = false;

    std::vector<std::vector<int64_t>> encoder_input_node_dims_;
    std::vector<const char*> encoder_input_node_names_;
    std::vector<const char*> encoder_output_node_names_;
    std::vector<std::vector<int64_t>> decoder_input_node_dims_;
    std::vector<const char*> decoder_input_node_names_;
    std::vector<const char*> decoder_output_node_names_;
    std::vector<std::vector<int64_t>> ctc_input_node_dims_;
    std::vector<const char*> ctc_input_node_names_;
    std::vector<const char*> ctc_output_node_names_;

public:
    WENET_DISALLOW_COPY_AND_ASSIGN(OnnxAsrModel);
};   

} //namespace

#endif