# EdgeML Lifecycle Management by LLMs

This project automates the repetitive stages of EdgeML application development using Large Language Models (LLMs). It can clean and engineer datasets, convert models to deployment formats, and generate sketches or Python runners for embedded and edge hardware.
 

## Installation

### Prerequisites

- Python 3.10 or 3.11 (required by `tensorflow==2.16.2`).
- `pip` and a virtual environment tool (`venv` or `conda`) to keep dependencies isolated.
- [Ollama](https://ollama.com/download) running locally if you plan to use self-hosted models, or OpenAI access if you configure `OPENAI_API_KEY`.
- [`arduino-cli`](https://arduino.github.io/arduino-cli/latest/installation/) if you intend to compile generated `.ino` sketches.
- SSH access to a Coral Dev Board (or compatible Edge TPU host) if you plan to run the TPU sketch generator. The project expects passwordless `ssh`/`scp` access.
- Optional: a Langfuse account or self-hosted instance for tracing.

### Steps

1. Clone the repository and enter the directory.

    ```bash
    git clone https://....git
    cd edgeml-autopilot-11.2025
    ```

2. Create and activate a virtual environment (optional but recommended).

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3. Install Python dependencies.

    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

4. Copy the example environment file and fill in the required values.

    ```bash
    cp example.env .env
    ```

    Key variables used in code:

    - `OPENAI_API_KEY`: required for OpenAI models through LiteLLM.
    - `OLLAMA_BASE_URL`: set if you run Ollama somewhere other than `http://localhost:11434`.
    - `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_HOST`: used for tracing when running against Langfuse Cloud.
    - `LANGFUSE_LOCAL_*`: alternate credentials for a self-hosted Langfuse instance.
    - `LANGCHAIN_API_KEY`: optional but still referenced for backwards compatibility.
    - `REMOTE_HOST`, `REMOTE_EXEC_PATH`, `REMOTE_PYTHON_ENV`, `REMOTE_PYTHON_EXECUTABLE`: required when running the TPU sketch generator because execution happens over SSH on the remote Coral device.

    Remember to restart your shell (or reload the environment) after updating `.env`.

5. Install the Arduino board support package used by the sketch generator.

    ```bash
    arduino-cli core update-index
    arduino-cli core install arduino:mbed_nano
    ```

    The project copies `compiling/model.h` into each build directory. Confirm the tensor in that header matches the model you deploy.

6. Install any Arduino libraries the generated sketches may request. The templates target `Arduino_APDS9960`, `Arduino_HTS221`, `Arduino_LSM9DS1`, `ArduinoBLE`, `Harvard_TinyMLx`, and `TensorFlowLite_ESP32`. Install the libraries you need via `arduino-cli lib install`.

 

## Usage

All commands assume you run them from the repository root.

### Run individual processors

The processors expose `__main__` blocks so you can launch each pipeline directly:

```bash
python src/processors/data_processor.py
python src/processors/model_converter.py
python src/processors/sketch_generator.py
python src/processors/pysketch_generator.py
python src/processors/tpusketch_generator.py
```

Each processor loads its default configuration (dataset paths, target boards, etc.) from hard-coded values inside the file. Adjust those values before running if you need different assets.

### Batch orchestration

`src/main.py` coordinates repeated runs across processors and models. Without CLI arguments it executes the `main()` function, which currently:

- Runs 30 iterations per processor (`data`, `convert`, `sketch`) by default.
- Iterates over the hard-coded `model_config_list` (Ollama models in the current version).
 

Before running batch tests:

1. Update `model_config_list` with the models and providers you have access to.
2. Reduce `num_runs` if you do not want 30 iterations.
 

You can also call `run_batch_test` manually:

```bash
python src/main.py <processor> <trace_id> <run_index> <benchmark_flag> <session_id> "('<provider>', '<model>', <parameters>)"
```

For example:

```bash
python src/main.py data 1234ab 1 False batch1 "('ollama', 'phi4:latest', False)"
```

The arguments mirror the signature in `main.run_batch_test`.

### Outputs

- Generated datasets and intermediate artifacts remain under `data/fruit_to_emoji/playground/`.
- Converted models appear in `models/fruit_to_emoji/tflite_model/`.
- Generated Arduino sketches compile inside `compiling/` and validated versions are stored under `results/object_detection/sketches/`.
- Logs for each processor are written to `logs/<Processor>.log`.
- TPU runs copy scripts into `/home/mendel/tinyml_autopilot/tmp` by default (configurable via `.env`).

## Source Code Structure

```text
src/
├── base/
│   ├── base_processor.py
│   │   └── llm_strategy.py
│   └── factories/
│       └── llm_factory.py
├── processors/
│   ├── data_processor.py
│   ├── model_converter.py
│   ├── pysketch_generator.py
│   ├── sketch_generator.py
│   └── tpusketch_generator.py
├── prompt_templates/
│   ├── TMPL_DP.py
│   ├── TMPL_MC.py
│   ├── TMPL_PSG.py
│   ├── TMPL_SG.py
│   └── TMPL_TPUSG.py
└── main.py
```

## Components

### Base layer

- `BaseProcessor` (`src/base/base_processor.py`): environment loading, Langfuse tracing, logging, remote execution helpers, code extraction, and Arduino compilation.
- `LLMStrategy` (`src/base/llm_strategy.py`): abstract strategy with LiteLLM-powered implementations for OpenAI and Ollama backends. Successful invocations are reported to Langfuse via LiteLLM callbacks.
- `LLMFactory` (`src/factories/llm_factory.py`): creates strategy instances for `"openai"` and `"ollama"` types.

### Processors

- `DataProcessor` (`src/processors/data_processor.py`): generates pandas transformations for the fruit RGB dataset, executes them in a playground directory, and retries on failure.
- `ModelConverter` (`src/processors/model_converter.py`): produces Python scripts that convert `model.keras` to TensorFlow Lite, optionally applying post-training quantization, and executes them.
- `SketchGenerator` (`src/processors/sketch_generator.py`): fills application specifications, injects programming guidelines, generates Arduino sketches, and compiles them with `arduino-cli`.
- `PySketchGenerator` (`src/processors/pysketch_generator.py`): creates Python runners for TFLite models (object detection use case) and validates them locally.
- `TPUSketchGenerator` (`src/processors/tpusketch_generator.py`): extends the Python sketch workflow to offload execution to a Coral Edge TPU host via SSH/SCP.

Each processor maintains its own log file, traces LLM calls with Langfuse, and captures the last generated code snippet when errors occur.

### Prompt templates

Prompt composition lives under `src/prompt_templates/`:

- `TMPL_DP.py`: suggestion tables and error handling for the data pipeline.
- `TMPL_MC.py`: conversion prompts for generating TensorFlow Lite scripts.
- `TMPL_SG.py`: spec filling and Arduino sketch generation instructions.
- `TMPL_PSG.py`: prompts for Python sketch generation.
- `TMPL_TPUSG.py`: prompt variants tailored for TPU-enabled Python scripts.

## Logging and Tracing

- Logs are written to `logs/<Processor>.log` and use `colorlog` for human-friendly console output.
- Langfuse traces are created for every processor run. Configure the appropriate `LANGFUSE_*` variables (cloud or self-hosted) in `.env`. Alternate credential sets (`LANGFUSE_LOCAL_*`, `LANGFUSE_LOCAL2_*`) are supported in code.
- Network latency is measured for each LLM invocation and attached as metadata in the Langfuse trace.

## Extending the Project

1. Create a new processor in `src/processors/` that subclasses `BaseProcessor`.
2. Provide prompt templates in `src/prompt_templates/` if the workflow requires new prompt variants.
3. Register the processor in orchestration code (for example, extend `processor_classes` in `main.py`).
4. Update `LLMFactory` if you need to support additional model providers.

Example skeleton:

```python
from base.base_processor import BaseProcessor

class NewProcessor(BaseProcessor):
     def get_user_input(self):
          # Populate fields used while composing prompts
          ...

     def run(self):
          self.get_user_input()
          # Compose prompts, invoke the LLM and execute results
          ...
```

## Debugging Steps

- Inspect `logs/<Processor>.log` for detailed error messages and the last generated code snippet.
- Verify `.env` values are loaded (`python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('OPENAI_API_KEY'))"`).
- Check LLM connectivity: `curl http://localhost:11434/api/tags` for Ollama or run a minimal LiteLLM call for OpenAI.
- Ensure `arduino-cli` can compile a known example sketch before running the sketch generator.
- For TPU runs, confirm passwordless SSH and that the remote Python environment contains required packages (`tensorflow`, `tflite-runtime`, `opencv-python`, etc.).
- If Langfuse traces do not appear, double-check the credential set (`LANGFUSE_*`, `LANGFUSE_LOCAL_*`, or `LANGFUSE_LOCAL2_*`).
 
 

## License

This project is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html). Ensure all redistributions comply with the GPLv3 terms.


