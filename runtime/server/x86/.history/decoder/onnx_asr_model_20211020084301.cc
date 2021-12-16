// Author lk9171@gmail.com

#include "decoder/onnx_asr_model.h"

#include <memory>
#include <utility>

namespace wenet {

void OnnxAsrModel::Read(
    const std::string& torch_model_path,
    const std::string& encoder_model_path,
    const std::string& ctc_model_path,
    const std::string& decoder_model_path,
    const int num_threads) {
        // init parameters
        torch::jit::script::Module model = torch::jit::load(torch_model_path);
        torch::jit::IValue o1 = model->run_method("subsampling_rate");
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
        encoder = Ort::Session(env_encoder, encoder_model_path.c_str(), sessionOpts);
        encoder_module_ = std::make_shared<OnnxModule>(std::move(encoder));
        Ort::Env env_ctc(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "ctc");
        ctc = Ort::Session(env_ctc, ctc_model_path.c_str(), sessionOpts);
        ctc_module_ = std::make_shared<OnnxModule>(std::move(ctc));
        Ort::Env env_decoder(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "decoder");
        decoder = Ort::Session(env_decoder, decoder_model_path.c_str(), sessionOpts);
        decoder_module_ = std::make_shared<OnnxModule>(std::move(decoder));

        LOG(INFO) << "torch model info subsampling_rate " << subsampling_rate_
        << " right context " << right_context_ << " sos " << sos_ << " eos "
        << eos_ << " is bidirectional decoder "
        << is_bidirectional_decoder_ << " num threads "
        << num_threads;
}

} //namespace