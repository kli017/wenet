// Copyright 2020 Mobvoi Inc. All Rights Reserved.
// Author: binbinzhang@mobvoi.com (Binbin Zhang)
//         di.wu@mobvoi.com (Di Wu)
// 2021    lk9171@gmail.com (Kai Li)

#include "decoder/onnx_asr_decoder.h"

#include <ctype.h>

#include <algorithm>
#include <limits>
#include <utility>

#include "decoder/ctc_endpoint.h"
#include "utils/string.h"
#include "utils/timer.h"

namespace wenet {

OnnxAsrDecoder::OnnxAsrDecoder(
    std::shared_ptr<FeaturePipeline> feature_pipeline,
    std::shared_ptr<OnnxDecodeResource> resource, 
    const DecodeOptions& opts)
    : feature_pipeline_(std::move(feature_pipeline)),
      model_(resource->onnx_model),
      symbol_table_(resource->symbol_table),
      fst_(resource->fst),
      unit_table_(resource->unit_table),
      opts_(opts),
      ctc_endpointer_(new CtcEndpoint(opts.ctc_endpoint_config)) {
  if (opts_.reverse_weight > 0) {
    // Check if model has a right to left decoder
    CHECK(model_->is_bidirectional_decoder());
  }
  if (nullptr == fst_) {
    searcher_.reset(new CtcPrefixBeamSearch(opts.ctc_prefix_search_opts));
  } else {
    searcher_.reset(new CtcWfstBeamSearch(*fst_, opts.ctc_wfst_search_opts));
  }
  ctc_endpointer_->frame_shift_in_ms(frame_shift_in_ms());
  subsampling_cache_ = std::move(InitNullCache(subsampling_cache_dims));
  elayers_output_cache_ = std::move(InitNullCache(elayers_output_cache_dims));
  conformer_cnn_cache_ = std::move(InitNullCache(conformer_cnn_cache_dims));
  InitPostProcessing();
}

void OnnxAsrDecoder::InitPostProcessing() {
  fst::SymbolTableIterator iter(*symbol_table_);
  std::string space_symbol = kSpaceSymbol;
  while (!iter.Done()) {
    if (iter.Symbol().size() > space_symbol.size() &&
        std::equal(space_symbol.begin(), space_symbol.end(),
                   iter.Symbol().begin())) {
      wp_start_with_space_symbol_ = true;
      break;
    }
    iter.Next();
  }
}

Ort::Value OnnxAsrDecoder::InitNullCache(const std::vector<int64_t>& cache_dims) {
  int cache_len = 1;
  for (const auto& cache_dim: cache_dims) {
    cache_len *= cache_dim;
  }

  std::vector<float> cache(cache_len, 0.0f);
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value cache_onnx = Ort::Value::CreateTensor<float>(memory_info,
                                                          cache.data(),
                                                          cache.size(),
                                                          cache_dims.data(),
                                                          cache_dims.size());
  return cache_onnx;
}

void OnnxAsrDecoder::Reset() {
  start_ = false;
  result_.clear();
  offset_ = 1;
  num_frames_ = 0;
  global_frame_offset_ = 0;
  num_frames_in_current_chunk_ = 0;
  Ort::Value subsampling_cache_init{nullptr};
  Ort::Value elayers_output_cache_init{nullptr};
  Ort::Value conformer_cnn_cache_init{nullptr};
  subsampling_cache_ = std::move(subsampling_cache_init);
  elayers_output_cache_ = std::move(elayers_output_cache_init);
  conformer_cnn_cache_ = std::move(conformer_cnn_cache_init);
  subsampling_cache_ = InitNullCache(subsampling_cache_dims);
  elayers_output_cache_ = InitNullCache(elayers_output_cache_dims);
  conformer_cnn_cache_ = InitNullCache(conformer_cnn_cache_dims);
  encoder_outs_.clear();
  cached_feature_.clear();
  searcher_->Reset();
  feature_pipeline_->Reset();
  ctc_endpointer_->Reset();
}

void OnnxAsrDecoder::ResetContinuousDecoding() {
  global_frame_offset_ = num_frames_;
  start_ = false;
  result_.clear();
  offset_ = 1;
  num_frames_in_current_chunk_ = 0;
  Ort::Value subsampling_cache_init{nullptr};
  Ort::Value elayers_output_cache_init{nullptr};
  Ort::Value conformer_cnn_cache_init{nullptr};
  subsampling_cache_ = std::move(subsampling_cache_init);
  elayers_output_cache_ = std::move(elayers_output_cache_init);
  conformer_cnn_cache_ = std::move(conformer_cnn_cache_init);
  subsampling_cache_ = InitNullCache(subsampling_cache_dims);
  elayers_output_cache_ = InitNullCache(elayers_output_cache_dims);
  conformer_cnn_cache_ = InitNullCache(conformer_cnn_cache_dims);
  encoder_outs_.clear();
  cached_feature_.clear();
  searcher_->Reset();
  ctc_endpointer_->Reset();
}

DecodeState OnnxAsrDecoder::Decode() { return this->AdvanceDecoding(); }

void OnnxAsrDecoder::Rescoring() {
  // Do attention rescoring
  Timer timer;
  AttentionRescoring();
  LOG(INFO) << "Rescoring cost latency: " << timer.Elapsed() << "ms.";
}

DecodeState OnnxAsrDecoder::AdvanceDecoding() {
  DecodeState state = DecodeState::kEndBatch;
  const int subsampling_rate = model_->subsampling_rate();
  const int right_context = model_->right_context();
  const int cached_feature_size = 1 + right_context - subsampling_rate;
  const int feature_dim = feature_pipeline_->feature_dim();
  int num_requried_frames = 0;
  // If opts_.chunk_size > 0, streaming case, read feature chunk by chunk
  // otherwise, none streaming case, read all feature at once
  if (opts_.chunk_size > 0) {
    if (!start_) {                      // First batch
      int context = right_context + 1;  // Add current frame
      num_requried_frames = (opts_.chunk_size - 1) * subsampling_rate + context;
    } else {
      num_requried_frames = opts_.chunk_size * subsampling_rate;
    }
  } else {
    num_requried_frames = std::numeric_limits<int>::max();
  }
  std::vector<std::vector<float>> chunk_feats;
  // If not okay, that means we reach the end of the input
  if (!feature_pipeline_->Read(num_requried_frames, &chunk_feats)) {
    state = DecodeState::kEndFeats;
  }

  num_frames_in_current_chunk_ = chunk_feats.size();
  num_frames_ += chunk_feats.size();
  LOG(INFO) << "Required " << num_requried_frames << " get "
            << chunk_feats.size();
  int num_frames = cached_feature_.size() + chunk_feats.size();
  // The total frames should be big enough to get just one output
  if (num_frames >= right_context + 1) {
    // 1. Prepare libtorch required data, splice cached_feature_ and chunk_feats
    // The first dimension is for batchsize, which is 1.

    std::vector<float> feats;
    // history feats
    for(auto e : cached_feature_)
        feats.insert(feats.end(), e.begin(), e.end());
    // new feats
    for(auto e : chunk_feats)
        feats.insert(feats.end(), e.begin(), e.end());
        

    LOG(INFO) << "feats: " << feats.size();
    LOG(INFO) << feats;
    

    std::vector<int64_t> input_mask_node_dims = {1, num_frames, feature_dim};

    //auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value onnx_feats = Ort::Value::CreateTensor<float>(memory_info, 
                                                            feats.data(), 
                                                            feats.size(), 
                                                            input_mask_node_dims.data(),
                                                            input_mask_node_dims.size());

    float *feats_onnx_1 = onnx_feats.GetTensorMutableData<float>();
    LOG(INFO) << "feat onnx: " << *feats_onnx_1 << "   " << *(feats_onnx_1 + 1);

    std::vector<int64_t> offset{offset_};
    std::vector<int64_t> offset_dim(0);
    Ort::Value onnx_offset = Ort::Value::CreateTensor<int64_t>(memory_info,
                                                              offset.data(),
                                                              offset.size(),
                                                              offset_dim.data(),
                                                              offset_dim.size());

    LOG(INFO) << "feat: "<< onnx_feats.GetTensorTypeAndShapeInfo().GetShape();
    LOG(INFO) << "offset: "<< onnx_offset.GetTensorTypeAndShapeInfo().GetShape();
    LOG(INFO) << "subsampling: "<< subsampling_cache_.GetTensorTypeAndShapeInfo().GetShape();
    LOG(INFO) << "elayers: "<< elayers_output_cache_.GetTensorTypeAndShapeInfo().GetShape();
    LOG(INFO) << "conformer: "<< conformer_cnn_cache_.GetTensorTypeAndShapeInfo().GetShape();

    std::vector<Ort::Value> input_onnx;
    input_onnx.emplace_back(std::move(onnx_feats));
    input_onnx.emplace_back(std::move(onnx_offset));
    input_onnx.emplace_back(std::move(subsampling_cache_));
    input_onnx.emplace_back(std::move(elayers_output_cache_));
    input_onnx.emplace_back(std::move(conformer_cnn_cache_));

    // directly model load test
    Ort::SessionOptions encoder_session_options_template;
    Ort::Env encoder_env_template(ORT_LOGGING_LEVEL_WARNING, "debug"); 
    std::string encoder_model_path_template = "/root/wenet-onnx/encoder_chunk_conformer.onnx";
    Ort::Session session(encoder_env_template, encoder_model_path_template.c_str(), encoder_session_options_template);

    std::vector<const char*> encoder_input_names = {"input", "offset", "i1", "i2", "i3"};
    std::vector<const char*> encoder_output_names = {"output", "o1", "o2", "o3"};

    auto encoder_onnx_outputs = session.Run(Ort::RunOptions{nullptr},
                                            encoder_input_names.data(),
                                            input_onnx.data(),
                                            input_onnx.size(),
                                            encoder_output_names.data(),
                                            encoder_output_names.size());
    
    const float* chunk_out_data = encoder_onnx_outputs.front().GetTensorData<const float>();
    LOG(INFO) << "encoder out: " << *chunk_out_data;

    Timer timer;
    // 2. Encoder chunk forward
    // auto encoder_onnx_outputs = model_->encoder_session()->Run( Ort::RunOptions{nullptr},
    //                                                             model_->encoder_input_node_names().data(),
    //                                                             input_onnx.data(),
    //                                                             input_onnx.size(),
    //                                                             model_->encoder_output_node_names().data(),
    //                                                             model_->encoder_output_node_names().size());
    CHECK_EQ(encoder_onnx_outputs.size(), 4);
    Ort::Value chunk_out = std::move(encoder_onnx_outputs[0]);
    subsampling_cache_ = std::move(encoder_onnx_outputs[1]); 
    elayers_output_cache_ = std::move(encoder_onnx_outputs[2]);
    conformer_cnn_cache_ = std::move(encoder_onnx_outputs[3]);
    offset_ += chunk_out.GetTensorTypeAndShapeInfo().GetShape()[1]; // ???

    float* t1 = subsampling_cache_.GetTensorMutableData<float>();
    float* t2 = elayers_output_cache_.GetTensorMutableData<float>();
    float* t3 = conformer_cnn_cache_.GetTensorMutableData<float>();
    LOG(INFO) << "subsamp out: " << *t1 << "  " << *(t1 + 1);
    LOG(INFO) << "elayer out: " << *t2 << "  " << *(t2 + 1);
    LOG(INFO) << "conformer out: " << *t3 << "  " << *(t3 + 1);
    
    // onnx encoder finished
    std::vector<int64_t> chunk_out_sizes = chunk_out.GetTensorTypeAndShapeInfo().GetShape();
    // for print
    int chunk_out_count = 1;
    for (auto& chunk_out_size : chunk_out_sizes) { 
      chunk_out_count *= chunk_out_size;
    }
    float* chunk_out_float = chunk_out.GetTensorMutableData<float>();
    std::vector<float> chunk_out_vector(chunk_out_count);
    FILE *fout=fopen("wenet_encoder_result.txt","w");
    for ( int i = 0; i < (chunk_out_count); ++i) {
      chunk_out_vector[i] = *chunk_out_float;
      fprintf(fout,"%f\n",*chunk_out_float);
      chunk_out_float = chunk_out_float + 1;
    }
    fclose(fout);
    ////////////////


    torch::Tensor chunk_out_tensor = torch::from_blob(chunk_out_float, 
                      {chunk_out_sizes[0], chunk_out_sizes[1], chunk_out_sizes[2]}, at::kFloat);
    
    encoder_outs_.emplace_back(std::move(chunk_out_tensor));  /// ???

    // The first dimension of returned value is for batchsize, which is 1
    // ctc
    auto ctc_onnx_outputs = model_->ctc_session()->Run(Ort::RunOptions{nullptr},
                                                  model_->ctc_input_node_names().data(),
                                                  &chunk_out, 
                                                  1,
                                                  model_->ctc_output_node_names().data(),
                                                  1);
    Ort::Value ctc_onnx_output_probs = std::move(ctc_onnx_outputs[0]);

    //auto type_info_encoder = chunk_out.GetTensorTypeAndShapeInfo();
    //size_t total_len = type_info.GetElementCount();

    std::vector<int64_t> ctc_probs_sizes = ctc_onnx_output_probs.GetTensorTypeAndShapeInfo().GetShape();
    float* ctc_probs_float = ctc_onnx_output_probs.GetTensorMutableData<float>();
    torch::Tensor ctc_log_probs = torch::from_blob(ctc_probs_float, 
                      {ctc_probs_sizes[0], ctc_probs_sizes[1], ctc_probs_sizes[2]}, at::kFloat);
    

    int forward_time = timer.Elapsed();
    timer.Reset();
    searcher_->Search(ctc_log_probs);
    int search_time = timer.Elapsed();
    VLOG(3) << "forward takes " << forward_time << " ms, search takes "
            << search_time << " ms";
    UpdateResult();

    if (ctc_endpointer_->IsEndpoint(ctc_log_probs, DecodedSomething())) {
      LOG(INFO) << "Endpoint is detected at " << num_frames_;
      state = DecodeState::kEndpoint;
    }

    // 3. Cache feature for next chunk
    if (state == DecodeState::kEndBatch) {
      // TODO(Binbin Zhang): Only deal the case when
      // chunk_feats.size() > cached_feature_size_ here, and it's consistent
      // with our current model, refine it later if we have new model or
      // new requirements
      CHECK(chunk_feats.size() >= cached_feature_size);
      cached_feature_.resize(cached_feature_size);
      for (int i = 0; i < cached_feature_size; ++i) {
        cached_feature_[i] = std::move(
            chunk_feats[chunk_feats.size() - cached_feature_size + i]);
      }
    }
  }

  start_ = true;
  return state;
}

void OnnxAsrDecoder::UpdateResult(bool finish) {
  const auto& hypotheses = searcher_->Outputs();
  const auto& inputs = searcher_->Inputs();
  const auto& likelihood = searcher_->Likelihood();
  const auto& times = searcher_->Times();
  result_.clear();

  CHECK_EQ(hypotheses.size(), likelihood.size());
  for (size_t i = 0; i < hypotheses.size(); i++) {
    const std::vector<int>& hypothesis = hypotheses[i];

    DecodeResult path;
    bool is_englishword_prev = false;
    path.score = likelihood[i];
    int offset = global_frame_offset_ * feature_frame_shift_in_ms();
    for (size_t j = 0; j < hypothesis.size(); j++) {
      std::string word = symbol_table_->Find(hypothesis[j]);
      if (wp_start_with_space_symbol_) {
        path.sentence += word;
        continue;
      }
      bool is_englishword_now = CheckEnglishWord(word);
      if (is_englishword_prev && is_englishword_now) {
        path.sentence += (' ' + word);
      } else {
        path.sentence += (word);
      }
      is_englishword_prev = is_englishword_now;
    }

    // TimeStamp is only supported in final result
    // TimeStamp of the output of CtcWfstBeamSearch may be inaccurate due to
    // various FST operations when building the decoding graph. So here we use
    // time stamp of the input(e2e model unit), which is more accurate, and it
    // requires the symbol table of the e2e model used in training.
    if (unit_table_ != nullptr && finish) {
      const std::vector<int>& input = inputs[i];
      const std::vector<int>& time_stamp = times[i];
      CHECK_EQ(input.size(), time_stamp.size());
      for (size_t j = 0; j < input.size(); j++) {
        std::string word = unit_table_->Find(input[j]);
        int start = j > 0 ? ((time_stamp[j - 1] + time_stamp[j]) / 2 *
                             frame_shift_in_ms())
                          : 0;
        int end = j < input.size() - 1 ? ((time_stamp[j] + time_stamp[j + 1]) /
                                          2 * frame_shift_in_ms())
                                       : offset_ * frame_shift_in_ms();
        WordPiece word_piece(word, offset + start, offset + end);
        path.word_pieces.emplace_back(word_piece);
      }
    }
    path.sentence = ProcessBlank(path.sentence);
    result_.emplace_back(path);
  }

  if (DecodedSomething()) {
    VLOG(1) << "Partial CTC result " << result_[0].sentence;
  }
}

float OnnxAsrDecoder::AttentionDecoderScore(const torch::Tensor& prob,
                                             const std::vector<int>& hyp,
                                             int eos) {
  float score = 0.0f;
  auto accessor = prob.accessor<float, 2>();
  for (size_t j = 0; j < hyp.size(); ++j) {
    score += accessor[j][hyp[j]];
  }
  score += accessor[hyp.size()][eos];
  return score;
}

void OnnxAsrDecoder::AttentionRescoring() {
  searcher_->FinalizeSearch();
  UpdateResult(true);
  // No need to do rescoring
  if (0.0 == opts_.rescoring_weight) {
    return;
  }
  // No encoder output
  if (encoder_outs_.size() == 0) {
    return;
  }

  int sos = model_->sos();
  int eos = model_->eos();
  // Inputs() returns N-best input ids, which is the basic unit for rescoring
  // In CtcPrefixBeamSearch, inputs are the same to outputs
  const auto& hypotheses = searcher_->Inputs();
  int num_hyps = hypotheses.size();
  if (num_hyps <= 0) {
    return;
  }

  
  // // Step 1: Prepare input for onnx
  std::vector<int64_t> hyps_length(num_hyps);
  int max_hyps_len = 0;
  for (size_t i = 0; i < num_hyps; ++i) {
    int length = hypotheses[i].size() + 1;
    max_hyps_len = std::max(length, max_hyps_len);
    hyps_length[i] = static_cast<int64_t>(length);
  }

  std::vector<int64_t> hyps_tensor(num_hyps*max_hyps_len);
  for (size_t i = 0; i < num_hyps; ++i) {
    const std::vector<int>& hyp = hypotheses[i];
    int start_index = i*max_hyps_len;
    hyps_tensor[start_index] = sos;
    for (size_t j = 0; j < hyp.size(); ++j) {
      hyps_tensor[start_index+j+i] = hyp[j];
    }
    for (size_t j = hyp.size() + 1; j < max_hyps_len; ++j) {
      hyps_tensor[start_index + j] = eos;
    }
  }
 

  // // Step 2: Forward attention decoder by hyps and corresponding encoder_outs_
  auto decoder_memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  LOG(INFO) << encoder_outs_[0].sizes();
  torch::Tensor encoder_out = torch::cat(encoder_outs_, 1);
  LOG(INFO) << encoder_out.sizes();

  std::vector<int64_t> broad_size={num_hyps,1,1};
  torch::Tensor encoder_out_broad=encoder_out.repeat(broad_size);
  int subsample_len = encoder_out_broad.size(1);
  LOG(INFO) << encoder_out_broad.sizes();

  std::vector<uint8_t> encoder_mask;
  int encoder_mask_size = 1 * num_hyps * subsample_len;
  for (int i = 0; i < encoder_mask_size; ++i) {
    encoder_mask.emplace_back((uint8_t)(true));
  } 

  std::vector<int64_t> decoder_input_node_dims{num_hyps, 1, encoder_out_broad.size(1)}; // beam, batch, frame

  Ort::Value encoder_mask_onnx = Ort::Value::CreateTensor<bool>(decoder_memory_info,
                                      reinterpret_cast<bool *>(encoder_mask.data()),
                                      encoder_mask_size,
                                      decoder_input_node_dims.data(),
                                      decoder_input_node_dims.size());

  std::vector<float> encoder_out_vector(encoder_out_broad.data_ptr<float>(),
      encoder_out_broad.data_ptr<float>() + encoder_out_broad.numel());
  LOG(INFO) << "encoder_out_vector size: " << encoder_out_vector.size();
  LOG(INFO) << "encoder_out_broad numel: " << encoder_out_broad.numel();     

  std::vector<int64_t> encoder_out_onnx_dims{num_hyps, encoder_out.size(1), 256}; // beam, chunk, outsize
  Ort::Value encoder_out_onnx = Ort::Value::CreateTensor<float>(decoder_memory_info,
                                                                  encoder_out_vector.data(),
                                                                  encoder_out_vector.size(),
                                                                  encoder_out_onnx_dims.data(),
                                                                  encoder_out_onnx_dims.size());

  std::vector<int64_t> hyps_tensor_dim = {num_hyps, max_hyps_len};
  Ort::Value hyps_tensor_onnx = Ort::Value::CreateTensor<int64_t>(decoder_memory_info,
                                                                  hyps_tensor.data(),
                                                                  hyps_tensor.size(),
                                                                  hyps_tensor_dim.data(),
                                                                  hyps_tensor_dim.size());

  std::vector<int64_t> hyps_length_dim = {num_hyps};
  Ort::Value hyps_lens_onnx = Ort::Value::CreateTensor<int64_t>(decoder_memory_info,
                                                                hyps_length.data(), 
                                                                hyps_length.size(), 
                                                                hyps_length_dim.data(), 
                                                                hyps_length_dim.size()); 
  std::vector<Ort::Value> decoder_inputs;
  decoder_inputs.emplace_back(std::move(encoder_out_onnx));
  decoder_inputs.emplace_back(std::move(encoder_mask_onnx));
  decoder_inputs.emplace_back(std::move(hyps_tensor_onnx));
  decoder_inputs.emplace_back(std::move(hyps_lens_onnx));

  const char* decoder_input_names[] = {"input", "encoder_mask", "hyps_pad", "hyps_lens"};
  const char* decoder_output_names[] = {"output", "o1", "olens"};

  Ort::SessionOptions decoder_session_options_template;
  Ort::Env decoder_env_template(ORT_LOGGING_LEVEL_WARNING, "d"); 
  std::string decoder_model_path_template = "/root/wenet-onnx/decoder.onnx";
  Ort::Session decoder_session_template(decoder_env_template, decoder_model_path_template.c_str(), decoder_session_options_template);

  auto decoder_outputs = decoder_session_template.Run(Ort::RunOptions{nullptr},
                                                      decoder_input_names,
                                                      decoder_inputs.data(),
                                                      decoder_inputs.size(),
                                                      decoder_output_names,
                                                      3);
  
  Ort::Value decoder_outputs_first = std::move(decoder_outputs[0]);

  std::vector<int64_t> decoder_outputs_first_sizes = decoder_outputs_first.GetTensorTypeAndShapeInfo().GetShape();
  int decoder_outputs_first_count = 1;
  for (auto &decoder_outputs_first_size : decoder_outputs_first_sizes) {
    decoder_outputs_first_count *= decoder_outputs_first_size;
  }

  std::vector<float> decoder_outputs_first_vector(decoder_outputs_first_count);
  float* decoder_outputs_first_float = decoder_outputs_first.GetTensorMutableData<float>();
  for (int i = 0; i < decoder_outputs_first_count - 1; ++i) {
    decoder_outputs_first_vector[i] = *decoder_outputs_first_float;
    decoder_outputs_first_float = decoder_outputs_first_float + 1;
  }
  torch::Tensor probs = torch::from_blob(decoder_outputs_first_vector.data(),
  {decoder_outputs_first_size[0], decoder_outputs_first_sizes[1], decoder_outputs_first_sizes[2]}, at::kFloat);

  CHECK_EQ(probs.size(0), num_hyps);
  CHECK_EQ(probs.size(1), max_hyps_len);
  //step 3: compute rescoring score
  for (size_t i = 0; i < num_hyps; ++i) {
    const std::vector<int>& hyp = hypotheses[i];
    float score = 0.0f;
    // left to right decoder score
    score = AttentionDecoderScore(probs[i], hyp, eos);
    // optional: used for right to left score
    float r_score = 0.0f;
    if (opts_reverse_weight > 0) {
      //right to left score
      CHECK_EQ(probs.size(0), num_hyps);
      CHECK_EQ(probs.size(1), max_hyps_len);
      std::vector<int> r_hyp(hyp.size());
      std::reverse_copy(hyp.begin(), hyp.end(), r_hyp.begin());
      // right to left decoder score
      r_score = AttentionDecoderScore(probs[i], r_hyp, eos);
    }
    // combined reverse attention score
    score = 
        (score * (1 - opts_.reverse_weight)) + (r_score * opts_.reverse_weight);
    // combined ctc score
    result_[i].score = 
        opts_.rescoring_weight * score + opts.ctc_weight * result_[i].score;
  }
  std::sort(result_.begin(), result_.end(), DecodeResult::CompareFunc);


} // attentionrescore


}  // namespace wenet
