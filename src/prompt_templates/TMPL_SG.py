context_pro_tem = """
You are an expert in Machine Learning on Edge Devices, highly skilled in the TinyML workflows, tools, techniques, and best practices. Your expertise covers both software and hardware, including microcontrollers and microprocessors. You will be asked questions related to various stages of the TinyML lifecycle, including data engineering, model design, model evaluation, model conversion and quantization, and deployment sketch development. The main task is to generate code to perform corresponding tasks (e.g., data cleaning, model quantization).

Your output will be executed **directly**, without additional checks or modifications. Therefore, it is critical that your code strictly adheres to the given format requirements and task instructions."""



format_spec_filling_pro_tem = r"""
### RESPONSE FORMAT ###
- Output MUST contain exactly one code block, and nothing else.
- The code block must include ONLY the updated `"application_specifications":{{...}}` object.
- <CRITICAL> Retain the placeholder `"programming_guidelines": {programming_guidelines}` exactly as provided.</CRITICAL>
- The output should be self-contained with no information loss.
"""

spec_filling_pro_tem = r"""
### OBJECTIVE ###
Fill in the requested fields of `application_specifications`.

### INSTRUCTIONS ###
- Read `"board_fullname"` from `"hardware"`.
- Based on board, application description, and sensors, fill in the placeholder fields:
  - `{{decide_when_generating_code_based_on_given_board_and_application_description}}` → values determined by board/application.
  - `{{decide_when_generating_code_based_on_given_data_sample_and_application_description}}`  → values determined by dataset/application.
- Keep `"programming_guidelines"` unchanged.
- Ensure libraries are compatible with `{board_fullname}`.
- Use `"TensorFlowLite.h"` instead of `"Arduino_TensorFlowLite.h"`.

### DATASET INFORMATION ###
**DATASET SUMMARY**: 
{dataset_summary}

### TARGET TO BE FILLED ###
```json
{app_spec_pro_tem}
```
"""

# placeholders: dataset_summary, board_fullname, executed_code, error_info, app_spec_pro_tem
# count_placeholders: 5
error_handling_spec_filling_pro_tem = r"""
### OBJECTIVE ###
Regenerate the `application_specifications` to avoid errors in the previously returned version.

### INSTRUCTIONS ###
-. Review the caused error and format rules.
-. If any required fields were missing, add them back.
- Read `"board_fullname"` from `"hardware"`.
- Based on board, application description, and sensors, fill in the placeholder fields:
  - `{{decide_when_generating_code_based_on_given_board_and_application_description}}` → values determined by board/application.
  - `{{decide_when_generating_code_based_on_given_data_sample_and_application_description}}`  → values determined by dataset/application.
- Keep `"programming_guidelines"` unchanged.
- Ensure libraries are compatible with `{board_fullname}`.
- Use `"TensorFlowLite.h"` instead of `"Arduino_TensorFlowLite.h"`.

### DATASET INFORMATION ###
**DATASET SUMMARY**: 
{dataset_summary}

### PREVIOUS RETURNED APPLICATION SPECIFICATIONS ###
```json
{executed_code}
```

### CAUSED ERROR ###
```
{error_info}
```

### TARGET TO BE FILLED ###
```json
{app_spec_pro_tem}
```
"""


format_code_gen_pro_tem = r"""
### RESPONSE FORMAT ###
- Output MUST contain EXACTLY ONE code block with valid sketch code in C++ (.ino format).
- <CRITICAL>The sketch MUST include the path to the model file: `#include "model.h"`.</CRITICAL>
- If any detail is uncertain, skip it rather than guessing.
- Only code is needed in the output
- Adding anything outside the single code block will break the program.
"""

# placeholders: dataset_summary, app_spec_pro_tem
# count_placeholders: 2
code_gen_pro_tem = r"""
### OBJECTIVE ###
Generate an Arduino .ino sketch in C++ for the described application.

### INSTRUCTIONS ###
- Follow the `programming_guidelines` property from the `application_specification`, which is a programming guidelines for arduino sketches.
- Include `#include "model.h"`.
- Ensure the sketch is clear, correct, and executable.

### DATASET INFORMATION ###
- **DATASET SUMMARY**: {dataset_summary} 

### APPLICATION SPECIFICATIONS ###
```json
{app_spec_pro_tem}
```

### OUTPUT TEMPLATE ###
```cpp
<complete_sketch_code>
```
"""

# placeholders: dataset_summary, executed_code, error_info, app_spec_pro_tem
# count_placeholders: 4
error_handling_code_gen_pro_tem = r"""
### OBJECTIVE ###
Regenerate the .ino sketch code for the application to avoid the error shown below.

### INSTRUCTIONS ###
- Follow the `programming_guidelines` property from the `application_specification`, which is a programming guidelines for arduino sketches.
- Modify ONLY what is necessary to resolve the error.
- Ensure the sketch compiles for `{board_fullname}`.
- Always include: `#include "model.h"`.

### EXECUTED CODE BEFORE ###
```cpp
{executed_code}
```

### CAUSED ERROR BEFORE ###
```
{error_info}
```

### DATASET INFORMATION ###
- **DATASET SUMMARY**: {dataset_summary}

### APPLICATION SPECIFICATIONS ###
```json
{app_spec_pro_tem}
```

### OUTPUT TEMPLATE ###
```cpp
<complete_sketch_code>
```
"""

app_spec_pro_tem = r"""{{
"application_specifications": {{
    "application": {{
        "name": "{application_name_hinting_its_purpose}",
        "description": "{application_description}"}},
    "hardware": {{
        "board": "{board_fullname}",
        "sensors": {{"{decide_when_generating_code_based_on_given_board_and_application_description}": "{decide_when_generating_code_based_on_given_board_and_application_description}"}}}},
    "software": {{
        "libraries": {{
            "main_library": {{
                "name": "TensorFlowLite",
                "header": "TensorFlowLite.h"}},
            "other_libraries": [
                {{
                    "name": "{decide_when_generating_code_based_on_given_board_and_application_description}",
                    "header": "{decide_when_generating_code_based_on_given_board_and_application_description}.h"
                }},
                {{
                    "name": "{decide_when_generating_code_based_on_given_board_and_application_description}",
                    "header": "{decide_when_generating_code_based_on_given_board_and_application_description}.h"
                }},
                {{{decide_when_generating_code_based_on_given_board_and_application_description}}},
                {decide_when_generating_code_based_on_given_board_and_application_description}]
        }},
        "model": {{
            "path": "./model.h",
            "input_tensor": {{
                "dimensions": "{decide_when_generating_code_based_on_given_data_sample_and_application_description}",
                "data_type": "{input_datatype}"
            }},
            "output_tensor": {{
                "dimensions": "{decide_when_generating_code_based_on_given_data_sample_and_application_description}",
                "data_type": "{output_datatype}"
            }},
            "tensor_arena_size": "{decide_when_generating_code_based_on_given_data_sample_and_application_description}"
        }}
    }},
    "deployment": {{
        "device": "{board_fullname}",
        "communication_interface": "Serial",
        "baud_rate": 9600
    }},
    "classification": {{
        "classes": {classification_classes}
    }},
    "programming_guidelines": {programming_guidelines},}}}}"""

# Programming Guidelines template outlining necessary steps for sketch code
programming_guidelines_pro_tem = r"""{
"Programming Guidelines for TFLite Micro Inference on Microcontrollers": {
"Phase 1": {
  "Initialization": {
    "1.1": {"Include Necessary Libraries": "Include all necessary libraries according to the application and target board. IMPORTANT: Always include the base library (`TensorFlowLite.h`) before dependent headers (`tensorflow/lite/micro/*.h`) since dependent libraries rely on core definitions from the base library. Also include sensor libraries, I/O libraries, or other application-specific dependencies."},

    "1.2": {"Declare Variables": "Declare critical variables for TensorFlow Lite Micro, including at least: `ErrorReporter` (logs errors), `ModelPointer` (holds the flat buffer model), `Interpreter` (executes inference), and `InputPointer` (points to input tensor). Declare other application-specific variables as needed."},
    "1.3": {"Define Tensor Arena": "Define and allocate a tensor arena buffer with a carefully chosen size. Ensure sufficient memory is available to prevent crashes while avoiding unnecessary waste."},
    "1.4": {"Load the Model": "Load the model from the given path using the provided TFLite Micro function and check for validity."},
    "1.5": {"Resolve Operators": "Register the necessary operators using an `OpResolver`. Use specific operators if model architecture is known. Otherwise, use `AllOpsResolver` as a fallback."},
    "1.6": {"Instantiate the Interpreter": "Initialize the interpreter with the model, operator resolver, tensor arena, and error reporter."},
    "1.7": {"Allocate Memory": "Allocate memory for tensors using the tensor arena. Perform error checking to confirm success."},
    "1.8": {"Define Model Inputs": "Retrieve model input tensor details (dimensions, dtype). Ensure compatibility with the application's input data."},
    "1.9": {"Set Up Other Relevant Parts": "Initialize other non-ML components required by the application (e.g., peripherals, communication modules, actuators)."}
  }
},
"Phase 2": {
  "Preprocessing": {
    "2.1": {"Sensor Setup": "Initialize and configure the sensor(s) or input data source(s). Ensure data acquisition works correctly."},
    "2.2": {"Optional Feature Extraction": "If needed, apply transformations or feature extraction on raw sensor data before feeding it to the model."}
  }
},
"Phase 3": {
  "Inference": {
    "3.1": {"Data Copy": "Copy the processed data (from sensor buffer or extracted features) into the model's input tensor buffer."},
    "3.2": {"Invoke Interpreter": "Run the inference using `interpreter.Invoke()`. Check return codes if applicable."}
  }
},
"Phase 4": {
  "Postprocessing": {
    "4.1": {"Process Output": "Interpret the model's output tensor(s) according to application requirements. Apply thresholds, mappings, or scaling if needed."},
    "4.2": {"Execute Application Behavior": "Trigger application-specific actions or behaviors based on the inference result (e.g., display output, send signal)."}
  }},}}
"""