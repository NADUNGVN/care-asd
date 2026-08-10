#include <cuda_runtime_api.h>
#include <NvInferRuntime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cerr << "[TensorRT] " << message << '\n';
    }
  }
};

template <typename T>
struct TrtDeleter {
  void operator()(T* object) const noexcept {
    if (object != nullptr) {
      object->destroy();
    }
  }
};

using Clock = std::chrono::steady_clock;

struct Options {
  std::string engine_path;
  std::string input_path;
  std::string output_path;
  std::string report_path;
  std::size_t input_elements{0};
  int warmup{100};
  int iterations{1000};
};

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}

void check_cuda(cudaError_t result, const std::string& action) {
  if (result != cudaSuccess) {
    fail(action + ": " + cudaGetErrorString(result));
  }
}

int parse_positive_int(const std::string& value, const std::string& flag, bool allow_zero) {
  try {
    const auto parsed = std::stoll(value);
    if (parsed < 0 || (!allow_zero && parsed == 0)) {
      fail(flag + " must be " + (allow_zero ? "non-negative" : "positive"));
    }
    return static_cast<int>(parsed);
  } catch (const std::invalid_argument&) {
    fail(flag + " must be an integer");
  } catch (const std::out_of_range&) {
    fail(flag + " is outside the supported range");
  }
}

std::size_t parse_elements(const std::string& value) {
  try {
    const auto parsed = std::stoull(value);
    if (parsed == 0) {
      fail("--input-elements must be positive");
    }
    return static_cast<std::size_t>(parsed);
  } catch (const std::invalid_argument&) {
    fail("--input-elements must be an unsigned integer");
  } catch (const std::out_of_range&) {
    fail("--input-elements is outside the supported range");
  }
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; index += 2) {
    if (index + 1 >= argc) {
      fail("missing value for " + std::string(argv[index]));
    }
    const std::string flag = argv[index];
    const std::string value = argv[index + 1];
    if (flag == "--engine") {
      options.engine_path = value;
    } else if (flag == "--input") {
      options.input_path = value;
    } else if (flag == "--input-elements") {
      options.input_elements = parse_elements(value);
    } else if (flag == "--output") {
      options.output_path = value;
    } else if (flag == "--warmup") {
      options.warmup = parse_positive_int(value, flag, true);
    } else if (flag == "--iterations") {
      options.iterations = parse_positive_int(value, flag, false);
    } else if (flag == "--report") {
      options.report_path = value;
    } else {
      fail("unknown argument: " + flag);
    }
  }
  if (options.engine_path.empty() || options.input_path.empty() || options.output_path.empty() ||
      options.report_path.empty() || options.input_elements == 0) {
    fail("required: --engine --input --input-elements --output --report");
  }
  return options;
}

std::vector<char> read_bytes(const std::string& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) {
    fail("cannot open file: " + path);
  }
  const auto size = stream.tellg();
  if (size <= 0) {
    fail("file is empty: " + path);
  }
  std::vector<char> bytes(static_cast<std::size_t>(size));
  stream.seekg(0);
  stream.read(bytes.data(), size);
  if (!stream) {
    fail("failed to read file: " + path);
  }
  return bytes;
}

std::vector<float> read_float32(const std::string& path, std::size_t elements) {
  const auto bytes = read_bytes(path);
  const auto expected = elements * sizeof(float);
  if (bytes.size() != expected) {
    fail("input byte length does not match --input-elements");
  }
  std::vector<float> values(elements);
  std::memcpy(values.data(), bytes.data(), expected);
  return values;
}

void write_float32(const std::string& path, const std::vector<float>& values) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) {
    fail("cannot open output file: " + path);
  }
  stream.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
  if (!stream) {
    fail("failed to write output file: " + path);
  }
}

std::size_t volume(const nvinfer1::Dims& dimensions) {
  std::size_t result = 1;
  for (int index = 0; index < dimensions.nbDims; ++index) {
    if (dimensions.d[index] <= 0) {
      fail("runner only accepts fixed, positive tensor dimensions");
    }
    result *= static_cast<std::size_t>(dimensions.d[index]);
  }
  return result;
}

double percentile(std::vector<double> samples, double quantile) {
  std::sort(samples.begin(), samples.end());
  const auto index = static_cast<std::size_t>(
      std::ceil(quantile * static_cast<double>(samples.size() - 1)));
  return samples.at(index);
}

void write_report(const std::string& path, const Options& options,
                  const std::vector<double>& durations_ms, std::size_t output_elements) {
  std::ofstream stream(path, std::ios::trunc);
  if (!stream) {
    fail("cannot open report file: " + path);
  }
  const double sum = std::accumulate(durations_ms.begin(), durations_ms.end(), 0.0);
  stream << std::fixed << std::setprecision(6);
  stream << "{\n"
         << "  \"runner\": \"care_asd_trt_runner\",\n"
         << "  \"warmup_windows\": " << options.warmup << ",\n"
         << "  \"timed_windows\": " << options.iterations << ",\n"
         << "  \"input_elements\": " << options.input_elements << ",\n"
         << "  \"output_elements\": " << output_elements << ",\n"
         << "  \"model_latency_ms\": {\n"
         << "    \"mean\": " << sum / static_cast<double>(durations_ms.size()) << ",\n"
         << "    \"p50\": " << percentile(durations_ms, 0.50) << ",\n"
         << "    \"p95\": " << percentile(durations_ms, 0.95) << ",\n"
         << "    \"p99\": " << percentile(durations_ms, 0.99) << "\n"
         << "  }\n"
         << "}\n";
  if (!stream) {
    fail("failed to write report file: " + path);
  }
}

int run(const Options& options) {
  Logger logger;
  const auto engine_bytes = read_bytes(options.engine_path);
  const auto input = read_float32(options.input_path, options.input_elements);

  std::unique_ptr<nvinfer1::IRuntime, TrtDeleter<nvinfer1::IRuntime>> runtime(
      nvinfer1::createInferRuntime(logger));
  if (!runtime) {
    fail("failed to create TensorRT runtime");
  }
  std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter<nvinfer1::ICudaEngine>> engine(
      runtime->deserializeCudaEngine(engine_bytes.data(), engine_bytes.size()));
  if (!engine) {
    fail("failed to deserialize TensorRT engine");
  }
  std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter<nvinfer1::IExecutionContext>> context(
      engine->createExecutionContext());
  if (!context) {
    fail("failed to create TensorRT execution context");
  }

  int input_binding = -1;
  int output_binding = -1;
  const int binding_count = engine->getNbBindings();
  for (int binding = 0; binding < binding_count; ++binding) {
    if (engine->bindingIsInput(binding)) {
      if (input_binding != -1) {
        fail("engine must expose exactly one input binding");
      }
      input_binding = binding;
    } else {
      if (output_binding != -1) {
        fail("engine must expose exactly one output binding");
      }
      output_binding = binding;
    }
  }
  if (input_binding == -1 || output_binding == -1) {
    fail("engine must expose exactly one input and one output binding");
  }
  if (engine->getBindingDataType(input_binding) != nvinfer1::DataType::kFLOAT ||
      engine->getBindingDataType(output_binding) != nvinfer1::DataType::kFLOAT) {
    fail("runner supports float32 input and output bindings only");
  }
  const auto expected_input_elements = volume(engine->getBindingDimensions(input_binding));
  const auto output_elements = volume(engine->getBindingDimensions(output_binding));
  if (expected_input_elements != options.input_elements) {
    fail("--input-elements does not match the engine input tensor volume");
  }

  std::vector<void*> bindings(static_cast<std::size_t>(binding_count), nullptr);
  check_cuda(cudaMalloc(&bindings[static_cast<std::size_t>(input_binding)],
                        options.input_elements * sizeof(float)),
             "cudaMalloc input");
  check_cuda(cudaMalloc(&bindings[static_cast<std::size_t>(output_binding)],
                        output_elements * sizeof(float)),
             "cudaMalloc output");

  cudaStream_t stream{};
  check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");
  try {
    check_cuda(cudaMemcpyAsync(bindings[static_cast<std::size_t>(input_binding)], input.data(),
                               input.size() * sizeof(float), cudaMemcpyHostToDevice, stream),
               "copy input to device");
    for (int iteration = 0; iteration < options.warmup; ++iteration) {
      if (!context->enqueueV2(bindings.data(), stream, nullptr)) {
        fail("TensorRT warm-up inference failed");
      }
    }
    check_cuda(cudaStreamSynchronize(stream), "synchronize warm-up");

    std::vector<double> durations_ms;
    durations_ms.reserve(static_cast<std::size_t>(options.iterations));
    for (int iteration = 0; iteration < options.iterations; ++iteration) {
      const auto started = Clock::now();
      if (!context->enqueueV2(bindings.data(), stream, nullptr)) {
        fail("TensorRT inference failed");
      }
      check_cuda(cudaStreamSynchronize(stream), "synchronize inference");
      const auto ended = Clock::now();
      durations_ms.push_back(std::chrono::duration<double, std::milli>(ended - started).count());
    }

    std::vector<float> output(output_elements);
    check_cuda(cudaMemcpyAsync(output.data(), bindings[static_cast<std::size_t>(output_binding)],
                               output.size() * sizeof(float), cudaMemcpyDeviceToHost, stream),
               "copy output to host");
    check_cuda(cudaStreamSynchronize(stream), "synchronize output");
    write_float32(options.output_path, output);
    write_report(options.report_path, options, durations_ms, output_elements);
  } catch (...) {
    cudaStreamDestroy(stream);
    cudaFree(bindings[static_cast<std::size_t>(input_binding)]);
    cudaFree(bindings[static_cast<std::size_t>(output_binding)]);
    throw;
  }
  check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
  check_cuda(cudaFree(bindings[static_cast<std::size_t>(input_binding)]), "cudaFree input");
  check_cuda(cudaFree(bindings[static_cast<std::size_t>(output_binding)]), "cudaFree output");
  return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(parse_options(argc, argv));
  } catch (const std::exception& error) {
    std::cerr << "care_asd_trt_runner: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
