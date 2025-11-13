context_pro_tem = """
You are an expert in Machine Learning on Edge Devices, highly skilled in the TinyML workflows, tools, techniques, and best practices. Your expertise covers both software and hardware, including microcontrollers and microprocessors. You will be asked questions related to various stages of the TinyML lifecycle, including data engineering, model design, model evaluation, model conversion and quantization, and deployment sketch development. The main task is to generate code to perform corresponding tasks (e.g., data cleaning, model quantization).

Your output will be executed **directly**, without additional checks or modifications. Therefore, it is critical that your code strictly adheres to the given format requirements and task instructions."""


format_pro_tem = r"""
### RESPONSE FORMAT ###
- Output must contain exactly one Python code block, with no text before or after.
- Code must be complete, clear, and directly executable.
- If quantization is required: 
  - Define `inference_input_type`, `inference_output_type`, and `supported_ops` strictly based on the specified `data_type`.
  - `supported_ops` must include TFLITE_BUILTINS and/or TFLITE_BUILTINS_INT8, depending on requirements.
- Variable and file naming should clearly reflect quantization configuration.
"""

code_gen_pro_tem = r"""
### OBJECTIVE ###
Convert the original model to TFLite{quant_req_e}.

### INSTRUCTIONS ###
- Generate complete, runnable Python code to perform the conversion to tflite, {quant_req_e}.
- Include necessary imports.
- Do not output explanations or text outside the code block.

### CONFIGURATION PARAMETERS ###
- **Data Type**: 
    - input_datatype= "{input_datatype}"
    - output_datatype= "{output_datatype}"

### PREDEFINED PATHS ###
- original_model_path="{original_model_path}"
- converted_model_path="{converted_model_path}"

### OUTPUT TEMPLATE ###
```python
<complete_code>
```
"""

error_handling_code_gen_pro_tem = r"""
### OBJECTIVE ###
Regenerate the model conversion code to avoid error reported in ### CAUSED ERROR ###.

### INSTRUCTIONS ###
- Carefully review the error and regenerate the code to avoid it.
- Ensure code is self-contained with all required imports.
- Implement the operation: Convert the original model to TFLite {quant_req_e}.
- Do not include explanations, comments, or error messages outside the code block.

### CONFIGURATION PARAMETERS ###
- **Data Type**: 
    - input_datatype= "{input_datatype}"
    - output_datatype= "{output_datatype}"

### PREDEFINED PATHS ###
- original_model_path="{original_model_path}"
- converted_model_path="{converted_model_path}"

### EXECUTED CODE ###
```python
{executed_code}
```

### CAUSED ERROR ###
```
{error_info}
```

### OUTPUT TEMPLATE ###
```python
<complete_code>
```
"""
