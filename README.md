3 
# TinyML Lifecycle Management By LLMs

This project automates the lifecycle management of TinyML applications using Large Language Models (LLMs). It streamlines the process of data processing, model conversion, and Arduino sketch generation for TinyML projects.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Components](#components)
- [Logging and Tracing](#logging-and-tracing)
- [Extending the Project](#extending-the-project)
- [Debugging Steps](#debugging-steps)
- [License](#license)
- [Known Issues](#known-issues)

## Installation

To set up this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://version.helsinki.fi/wuguangh/tinyml-autopilot.git
   cd tinyml-autopilot
   ````

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Rename `.env.example` to `.env` and fill in the needed keys.
   ```
   OPENAI_API_KEY=""
   OPENAI_MODEL_NAME=""
   LANGCHAIN_API_KEY=""
   ```
4. `arduino-cli` is required for sketch generation. Install it from [here](https://arduino.github.io/arduino-cli/latest/installation/).

    [Search](https://arduino.github.io/arduino-cli/0.21/commands/arduino-cli_core_search/) for the `arduino-cli core` coresponding to your board:
   ```bash
   # Search for a core in Boards Manager
   arduino-cli core search <keywords...> [flags]
   ```

    [Install](https://arduino.github.io/arduino-cli/0.21/commands/arduino-cli_core_install/) the core:
   ```bash
   # Install a core from a package manager
   arduino-cli core install PACKAGER:ARCH[@VERSION]... [flags]
   ```
   `Arduino Nano 33 BLE` as an example:
   ```bash
   arduino-cli core install arduino:mbed_nano
   ```
5. Installation of necessary and recommended `arduino-cli` libraries:  
   - Recommended: `Harvard_TinyMLx, Arduino_APDS9960, Arduino_HTS221, Arduino_LSM9DS1, ArduinoBLE, FreeRTOS, tcs3200, TensorFlowLite_ESP32`, etc.

## Usage

This project for now consists of three main components: Data Processor, Model Converter, and Sketch Generator. Here's how to use each:

All the three components can be run individually:

```bash
python data_processor.py
python model_converter.py
python sketch_generator.py
```

or run the `main.py`, which also supports batch testing:
```python
def main():
    # data: DataProcessor, convert: ModelConverter, sketch: SketchGenerator
    stage = "sketch"
    num_runs = 30
    benchmarking = num_runs >= 30
    ...
```

```bash
python main.py
```

### Demo: [Arduino Sketch Generation Demo](https://youtu.be/GsY5XDClerk?si=2FRgZS5EYmrs9LBh)

## Project Structure

```
src/
├── __init__.py
├── base/
│   ├── __init__.py
│   ├── base_processor.py
│   └── llm_strategy.py
├── factories/
│   ├── __init__.py
│   └── llm_factory.py
├── main.py
├── processors/
│   ├── __init__.py
│   ├── data_processor.py
│   ├── model_converter.py
│   └── sketch_generator.py
└── prompt_templates/
    ├── __init__.py
    ├── templates_convert.py
    ├── templates_data_proc.py
    └── templates_sketch.py
```

## Components
### Prompt Templates
Located in the `src/prompt_templates/` directory, prompt templates define the instructions and context provided to the LLM for generating specific code snippets. These templates ensure consistency and accuracy in the code generation process.

- **`templates_sketch.py`**: Contains templates for generating Arduino `.ino` sketches, including guidelines for initialization, preprocessing, inference, and postprocessing steps.
  
- **`templates_data_proc.py`**: Defines templates for data processing tasks, detailing operations like data engineering and error handling.
  
- **`templates_convert.py`**: Includes templates related to model conversion tasks, such as converting TensorFlow models to TensorFlow Lite format.

### Processors

Processors are the core components responsible for handling specific tasks by interacting with the LLM. Each processor inherits from the `BaseProcessor` class, ensuring consistent functionality across different tasks.

Usage examples can be found under `if __name__ == "__main__":` in the processor `.py` files.

#### Data Processor

**File**: `src/processors/data_processor.py`

**Purpose**: Automates data preprocessing tasks, generating Python scripts to handle data cleaning, transformation, and feature engineering.

**Key Functionalities**:
- **User Input Acquisition**: Specifies dataset paths and descriptions.
- **Suggestion Table Generation**: Uses LLM to suggest a sequence of data processing operations.
- **Operation Code Generation**: Generates and executes code for each suggested data processing step.
- **Error Handling**: Regenerates code upon encountering execution errors, ensuring robust data processing workflows.

#### Model Converter

**File**: `src/processors/model_converter.py`

**Purpose**: Automates the conversion of machine learning models from one format to another, such as converting TensorFlow models to TensorFlow Lite format with or without quantization.

**Key Functionalities**:
- **User Input Acquisition**: Specifies original and converted model paths, quantization requirements, and data types.
- **Conversion Code Generation**: Uses LLM to generate code for model conversion tasks.
- **Error Handling**: Attempts to regenerate conversion code upon encountering execution errors.
- **Execution**: Runs the generated conversion scripts to produce the desired model format.
 


#### Sketch Generator
**File**: `src/processors/sketch_generator.py`

**Purpose**: Generates Arduino `.ino` sketch files based on specified application requirements and hardware specifications.

**Key Functionalities**:
- **User Input Acquisition**: Gathers details like application name, description, board type, sensor data types, and classification classes.
- **Specification Filling**: Utilizes prompt templates to create detailed application specifications.
- **Sketch Generation**: Generates the Arduino sketch by formatting prompts with specifications and guidelines.
- **Error Handling**: Attempts to regenerate code upon encountering errors during execution or compilation.
- **Execution**: Compiles the generated sketch using `arduino-cli` to ensure validity.
 
### Base Classes

#### BaseProcessor

**File**: `src/base/base_processor.py`

**Purpose**: Serves as the foundational class for all processors, providing shared functionalities such as environment setup, logging, LLM invocation with tracing, code execution, and error handling.

**Key Functionalities**:
- **Environment Loading**: Loads environment variables using `dotenv`.
- **Logging Setup**: Configures logging with both console and file handlers using `colorlog`.
- **LLM Invocation**: Invokes the LLM with tracing metadata, including network latency measurements.
- **Code Execution**: Executes generated code snippets, handling both Python and Arduino sketches.
- **Error Handling**: Manages exceptions and integrates with LangSmith for tracing errors.

**Key Methods**:
- `invoke_llm_with_trace(prompt, invoke_subtask=None)`: Invokes the LLM with tracing.
- `execute_code(code, tmp_dir, sketch=False)`: Executes or compiles the provided code snippet.
- `run()`: Abstract method to be implemented by subclasses for executing processor-specific tasks.

#### LLMStrategy

**File**: `src/base/llm_strategy.py`

**Purpose**: Defines an abstract interface for different LLM strategies, allowing flexibility in using various LLM providers.

**Key Functionalities**:
- **Invoke LLM**: Abstract method to send prompts to the LLM and retrieve responses.
- **Get Endpoint URL**: Retrieves the API endpoint URL for tracking purposes.

**Concrete Implementations**:
- **OpenAIStrategy**: Implements `LLMStrategy` for OpenAI's GPT models.
- **OllamaStrategy**: Implements `LLMStrategy` for Ollama's models (To be fully implemented).
 

### Factories

#### LLMFactory

**File**: `src/factories/llm_factory.py`

**Purpose**: Provides a factory method to instantiate different `LLMStrategy` implementations based on the specified type.

**Key Functionalities**:
- **Create LLM**: Returns an instance of the desired `LLMStrategy` (e.g., OpenAI, Ollama).
 


## Workflow Overview

1. **Initialization**:
   - Load environment variables.
   - Initialize the appropriate `LLMStrategy` using `LLMFactory`.
   - Instantiate the desired processor (`SketchGenerator`, `DataProcessor`, or `ModelConverter`) with the LLM strategy and a unique `trace_id`.

2. **User Input Acquisition**:
   - Each processor gathers required inputs, such as application specifications, dataset paths, or model details.

3. **Prompt Composition**:
   - Processors use predefined prompt templates to create detailed instructions for the LLM, ensuring the generated code aligns with project requirements.

4. **LLM Invocation and Code Generation**:
   - The processor sends the composed prompts to the LLM, receives generated code snippets, and attempts to execute or compile them.
  
5. **Error Handling and Retries**:
   - If execution fails, the processor uses error-handling prompt templates to ask the LLM to regenerate and refine the code. This process repeats up to a specified number of retries.

6. **Execution and Validation**:
   - Successfully generated code is executed or compiled to ensure it meets the intended functionality.

7. **Logging and Tracing**:
   - Throughout the process, detailed logs are maintained, and tracing is integrated with LangSmith for monitoring and debugging.

## Logging and Tracing

The project employs robust logging and tracing mechanisms to monitor and debug processes.

### Logging

- **Configuration**: Set up using `colorlog` for colored console logs and standard formatting for file logs.
- **Location**: Log files are stored in the `logs/` directory, named after each processor class (e.g., `SketchGenerator.log`).
- **Usage**:
  - Informational messages, errors, and debugging information are logged throughout the processing steps.

### Tracing

- **Integration**: Utilizes LangSmith for tracing LLM invocations and other processing steps.
- **Metadata**: Includes details like network latency, trace IDs, and task-specific tags.
- **Benefits**:
  - Facilitates performance monitoring.
  - Aids in debugging by providing detailed trace information for each task and subtask.



## Extending the Project

The modular architecture allows for easy extension and integration of new processors or LLM strategies.

### Adding a New Processor

1. **Create a New Processor File**:
   - Add a new Python file in the `src/processors/` directory (e.g., `new_processor.py`).

2. **Inherit from `BaseProcessor`**:
   ```python
   from base.base_processor import BaseProcessor

   class NewProcessor(BaseProcessor):
       def get_task_name(self, short=False):
           return "new_processor"

       def get_user_input(self):
           # Define user input acquisition logic
           pass

       def run(self):
           # Define the processing workflow
           pass
   ```

3. **Implement Required Methods**:
   - `get_user_input()`: Gather necessary inputs.
   - `run()`: Define the processing steps using LLM invocations and code execution.

4. **Define Prompt Templates**:
   - Add relevant templates in `src/prompt_templates/` and update the processor to use them.

### Implementing a New LLM Strategy

1. **Create a New Strategy Class**:
   - Inherit from `LLMStrategy` and implement the abstract methods.
   ```python
   from base.llm_strategy import LLMStrategy

   class NewLLMStrategy(LLMStrategy):
       def __init__(self, ...):
           # Initialize the new LLM
           pass

       def invoke(self, prompt):
           # Define how to invoke the new LLM
           pass

       def get_endpoint_url(self) -> str:
           # Return the LLM's API endpoint
           pass
   ```

2. **Update `LLMFactory`**:
   ```python
   class LLMFactory:
       @staticmethod
       def create_llm(llm_type, **kwargs):
           if llm_type == "newllm":
               return NewLLMStrategy(...)
           # Existing conditions
   ```

3. **Use the New Strategy**:
   ```python
   llm_strategy = LLMFactory.create_llm("newllm", ...)
   ```


### Debugging Steps

1. **Check Logs**:
   - Navigate to the `logs/` directory and inspect the relevant log file for error details.

2. **Verify Environment Variables**:
   - Ensure all necessary variables are set in the `.env` file.

3. **Test LLM Connectivity**:
   - Use a simple test script to verify that the LLM can be invoked successfully.

4. **Validate Generated Code**:
   - Manually review the generated code snippets to identify syntax or logical errors.

5. **Review Tracing Information**:
   - Use LangSmith to analyze trace data for insights into processing bottlenecks or failures.



## License

Apache License 2.0


## Known Issues

- MacOS cannot perform model conversion due to the `Tensorflow Lite` library issue. Linux should work fine.
 