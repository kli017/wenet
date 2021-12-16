#include <iomanip>
#include <utility>

#include "torch/script.h"

#include "decoder/params.h"
#include "frontend/wav.h"
#include "utils/flags.h"
#include "utils/log.h"
#include "utils/string.h"
#include "utils/timer.h"
#include "utils/utils.h"


DEFINE_bool(simulate_streaming, false, "simulate streaming input");
DEFINE_string(wav_path, "", "single wave path");
DEFINE_string(wav_scp, "", "input wav scp");
DEFINE_string(result, "", "result output file");

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    google::InitGoogleLogging(argv[0]);

    auto decode_config = wenet::InitDecodeOptionsFromFlags();
    auto feature_config = wenet::InitFeaturePipelineConfigFromFlags();
    auto decode_resource = wenet::InitOnnxDecodeResourceFromFlags();

    std::cout<< decode_resource->onnx_model.right_context() << std::endl;
    //std::vector<const char*> encoder_input_node_names = 

    //wenet::OnnxAsrDecoder decoder(feature_pipeline, decode_resource,
    //                               *decode_config);

    return 0;
}


