// Author: lk9171@gmail.com)


#ifndef DECODER_ONNX_ASR_DECODER_H_
#define DECODER_ONNX_ASR_DECODER_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "fst/fstlib.h"
#include "fst/symbol-table.h"
#include "torch/script.h"
#include "torch/torch.h"

#include "decoder/ctc_endpoint.h"
#include "decoder/ctc_prefix_beam_search.h"
#include "decoder/ctc_wfst_beam_search.h"
#include "decoder/torch_asr_model.h"
#include "frontend/feature_pipeline.h"
#include "utils/utils.h"

namespace wenet {

using TorchModule = torch::jit::script::Module;

struct DecodeOptions {
  // chunk_size is the frame number of one chunk after subsampling.
  // e.g. if subsample rate is 4 and chunk_size = 16, the frames in
  // one chunk are 64 = 16*4
  int chunk_size = 16;
  int num_left_chunks = -1;

  // final_score = rescoring_weight * rescoring_score + ctc_weight * ctc_score;
  // rescoring_score = left_to_right_score * (1 - reverse_weight) +
  // right_to_left_score * reverse_weight
  // Please note the concept of ctc_scores in the following two search
  // methods are different.
  // For CtcPrefixBeamSearch, it's a sum(prefix) score
  // For CtcWfstBeamSearch, it's a max(viterbi) path score
  // So we should carefully set ctc_weight according to the search methods.
  float ctc_weight = 0.0;
  float rescoring_weight = 1.0;
  float reverse_weight = 0.0;
  CtcEndpointConfig ctc_endpoint_config;
  CtcPrefixBeamSearchOptions ctc_prefix_search_opts;
  CtcWfstBeamSearchOptions ctc_wfst_search_opts;
};

struct WordPiece {
  std::string word;
  int start = -1;
  int end = -1;

  WordPiece(std::string word, int start, int end)
      : word(std::move(word)), start(start), end(end) {}
};

struct DecodeResult {
  float score = -kFloatMax;
  std::string sentence;
  std::vector<WordPiece> word_pieces;

  static bool CompareFunc(const DecodeResult& a, const DecodeResult& b) {
    return a.score > b.score;
  }
};

enum DecodeState {
  kEndBatch = 0x00,  // End of current decoding batch, normal case
  kEndpoint = 0x01,  // Endpoint is detected
  kEndFeats = 0x02   // All feature is decoded
};

// 上面不用改
// DecodeResource is thread safe, which can be shared for multiple
// decoding threads
struct OnnxDecodeResource {
  //std::shared_ptr<TorchAsrModel> torch_model = nullptr;
  std::shared_ptr<OnnxAsrModel> onnx_model = nullptr;
  std::shared_ptr<fst::SymbolTable> symbol_table = nullptr;
  std::shared_ptr<fst::Fst<fst::StdArc>> fst = nullptr;
  std::shared_ptr<fst::SymbolTable> unit_table = nullptr;
};

// ONNX ASR decoder
class OnnxAsrDecoder {
 public:
  OnnxAsrDecoder(std::shared_ptr<FeaturePipeline> feature_pipeline,
                  std::shared_ptr<DecodeResource> resource,
                  const DecodeOptions& opts);

  DecodeState Decode();
  void Rescoring();
  void Reset();
  void ResetContinuousDecoding();
  bool DecodedSomething() const {
    return !result_.empty() && !result_[0].sentence.empty();
  }

  // This method is used for time benchmark
  int num_frames_in_current_chunk() const {
    return num_frames_in_current_chunk_;
  }
  int frame_shift_in_ms() const {
    return model_->subsampling_rate() *
           feature_pipeline_->config().frame_shift * 1000 /
           feature_pipeline_->config().sample_rate;
  }
  int feature_frame_shift_in_ms() const {
    return feature_pipeline_->config().frame_shift * 1000 /
           feature_pipeline_->config().sample_rate;
  }
  const std::vector<DecodeResult>& result() const { return result_; }

 private:
  void InitPostProcessing();
  // Return true if we reach the end of the feature pipeline
  DecodeState AdvanceDecoding();
  void AttentionRescoring();

  float AttentionDecoderScore(const torch::Tensor& prob,
                              const std::vector<int>& hyp, int eos);
  void UpdateResult(bool finish = false);


  std::shared_ptr<FeaturePipeline> feature_pipeline_;
  //std::shared_ptr<TorchAsrModel> model_;
  std::shared_ptr<OnnxAsrModel> model_;

  std::shared_ptr<fst::Fst<fst::StdArc>> fst_ = nullptr;
  // output symbol table
  std::shared_ptr<fst::SymbolTable> symbol_table_;
  // e2e unit symbol table
  std::shared_ptr<fst::SymbolTable> unit_table_ = nullptr;
  const DecodeOptions& opts_;
  // cache feature
  std::vector<std::vector<float>> cached_feature_;
  bool start_ = false;

  // word piece start with space symbol["▁" (U+2581)] or not
  bool wp_start_with_space_symbol_ = false;

  //torch::jit::IValue subsampling_cache_;
  // transformer/conformer encoder layers output cache
  //torch::jit::IValue elayers_output_cache_;
  //torch::jit::IValue conformer_cnn_cache_;
  //std::vector<torch::Tensor> encoder_outs_;
  std::vector<Ort::Value> encoder_outs_;

  Ort::Value subsampling_cache_;
  std::vector<int64_t> subsampling_cache_dims = {1, 1, 256}; 
  Ort::Value elayers_output_cache_;
  std::vector<int64_t> elayers_output_cache_dims = {12, 1, 1, 256};
  Ort::Value conformer_cnn_cache_;
  std::vector<int64_t> conformer_cnn_cache_dims = {12, 1, 256, 15};

  std::vector<float> subsampling_cache_vector;
  std::vector<float> elayers_output_cache_vector;
  std::vector<float> conformer_cnn_cache_vector;

  //Ort::Value subsampling_cache_ = std::move(init_null_cache(subsampling_cache_1_dims, subsampling_cache_vector));
  //Ort::Value elayers_output_cache_11 = std::move(init_null_cache(elayers_output_cache_1_dims, elayers_output_cache_vector));
  //Ort::Value conformer_cnn_cache_11 = std::move(init_null_cache(conformer_cnn_cache_1_dims, conformer_cnn_cache_vector));

  int offset_ = 0;  // offset
  // For continuous decoding
  int num_frames_ = 0;
  int global_frame_offset_ = 0;
  // elayers_output and conformer_cnn cache max size
  int max_cache_size_ = 400;

  std::unique_ptr<SearchInterface> searcher_;
  std::unique_ptr<CtcEndpoint> ctc_endpointer_;

  int num_frames_in_current_chunk_ = 0;
  std::vector<DecodeResult> result_;

 public:
  WENET_DISALLOW_COPY_AND_ASSIGN(OnnxAsrDecoder);
};

}  // namespace wenet

#endif  // DECODER_ONNX_ASR_DECODER_H_
