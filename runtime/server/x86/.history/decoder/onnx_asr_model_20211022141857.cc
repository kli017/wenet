// Author lk9171@gmail.com

#include "decoder/onnx_asr_model.h"

#include <memory>
#include <utility>

namespace wenet {

void OnnxAsrModel::init_dims_and_names() {
    get_dims_and_names(encoder_module_, encoder_input_node_dims_, 
                        encoder_input_node_names_, encoder_output_node_names_);
    get_dims_and_names(decoder_module_, decoder_input_node_dims_, 
                        decoder_input_node_names_, decoder_output_node_names_);
    get_dims_and_names(ctc_module_, ctc_input_node_dims_, 
                        ctc_input_node_names_, ctc_output_node_names_);
} 


void OnnxAsrModel::get_dims_and_names(std::shared_ptr<Ort::Session> session,  
                        std::vector<std::vector<int64_t>>& input_node_dims,
                        std::vector<const char*>& input_node_names,
                        std::vector<const char*>& output_node_names) {
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_input_nodes = session->GetInputCount();
    input_node_names.resize(num_input_nodes);
    input_node_dims.resize(num_input_nodes);
    for (int i = 0; i < num_input_nodes; ++i) {
        char* input_name = session->GetInputName(i, allocator);
        LOG(INFO) << "input_name " << i << input_name;
        input_node_names[i] = input_name;
    }
    size_t num_output_nodes = session->GetOutputCount();
    output_node_names.resize(num_output_nodes);
    for (int i = 0; i < num_output_nodes; ++i) {
        char* output_name = session->GetOutputName(i, allocator);
        output_node_names[i] = output_name;
    }
}

void OnnxAsrModel::Read(
    const std::string& torch_model_path,
    const std::string& encoder_model_path,
    const std::string& ctc_model_path,
    const std::string& decoder_model_path,
    const int num_threads) {
        // init parameters
        torch::jit::script::Module model = torch::jit::load(torch_model_path);
        module_ = std::make_shared<TorchModule>(std::move(model));
        module_->eval();
        torch::jit::IValue o1 = module_->run_method("subsampling_rate");
        CHECK_EQ(o1.isInt(), true);
        subsampling_rate_ = o1.toInt();
        torch::jit::IValue o2 = module_->run_method("right_context");
        CHECK_EQ(o2.isInt(), true);
        right_context_ = o2.toInt();
        torch::jit::IValue o3 = module_->run_method("sos_symbol");
        CHECK_EQ(o3.isInt(), true);
        sos_ = o3.toInt();
        torch::jit::IValue o4 = module_->run_method("eos_symbol");
        CHECK_EQ(o4.isInt(), true);
        eos_ = o4.toInt();
        torch::jit::IValue o5 = module_->run_method("is_bidirectional_decoder");
        CHECK_EQ(o5.isBool(), true);
        is_bidirectional_decoder_ = o5.toBool();
        // load onnx models
        Ort::SessionOptions sessionOpts;
        sessionOpts.SetIntraOpNumThreads(num_threads);
        Ort::Env env_encoder(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "encoder");
        Ort::Session encoder = Ort::Session(env_encoder, encoder_model_path.c_str(), sessionOpts);
        encoder_module_ = std::make_shared<OnnxModule>(std::move(encoder));
        Ort::Env env_ctc(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "ctc");
        Ort::Session ctc = Ort::Session(env_ctc, ctc_model_path.c_str(), sessionOpts);
        ctc_module_ = std::make_shared<OnnxModule>(std::move(ctc));
        Ort::Env env_decoder(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "decoder");
        Ort::Session decoder = Ort::Session(env_decoder, decoder_model_path.c_str(), sessionOpts);
        decoder_module_ = std::make_shared<OnnxModule>(std::move(decoder));
        init_dims_and_names();

        LOG(INFO) << "torch model info subsampling_rate " << subsampling_rate_
        << " right context " << right_context_ << " sos " << sos_ << " eos "
        << eos_ << " is bidirectional decoder "
        << is_bidirectional_decoder_ << " num threads "
        << num_threads;
}


} //namespace