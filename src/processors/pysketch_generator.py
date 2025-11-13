# src/processors/pysketch_generator.py
import ast

# %%
import os
import sys
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from pprint import pprint

# Ensure base and factories are in the path for direct execution
# Adjust path logic if necessary based on your project structure
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from dotenv import load_dotenv  # Needed for direct run env loading
from langchain_core.prompts import PromptTemplate
from litellm import APIConnectionError

import prompt_templates.TMPL_PSG as TMPL_PSG
from base.base_processor import BaseProcessor
from factories.llm_factory import LLMFactory  # Needed for direct run

# tmp_dir = os.path.join(os.path.dirname(__file__), "..", "..", "tmp")
# ddir = os.path.dirname(__file__)
# print(Path(os.path.dirname(__file__)).parents[1])

# print(os.path.dirname(__file__))
 
# %%
class PySketchGenerator(BaseProcessor):
    """
    PySketchGenerator generates Python scripts for running TFLite models
    using tflite-runtime/tflite, based on a provided application description.
    It is designed to be generic across tasks, inputs, and outputs.
    """

    def __init__(
        self, llm_strategy, trace_id, num_run=None, benchmark=False, session_id=None, self_host_langfuse_flag=False
    ):
        """Initializes the PySketchGenerator."""
        super().__init__(
            llm_strategy,
            trace_id=trace_id,
            task_name=self.get_task_name(),  # Uses class name PySketchGenerator now
            num_run=num_run,
            benchmark=benchmark,
            session_id=session_id,
            self_host_langfuse_flag=self_host_langfuse_flag,
        )
        # Initialize specific attributes
        self.application_name = ""
        self.application_description = ""
        self.model_path = ""
        # Optional
        self.label_path = None
        self.input_description = ""
        self.input_path = None
        self.output_description = ""
        self.output_path = ""
        self.confidence_threshold = 0.5
        self.target_device = ""

        self.generated_sketch = ""

    def get_user_input(self):
        """
        Acquires and sets the user input parameters required for Python sketch generation.
        """
        # Example Task Definition:
        # self.confidence_threshold = 0.5
        # self.target_device = "Raspberry Pi 4B"
        # self.application_name = "Object Detection via images"
        # self.application_description = "This application uses a TFLite model ssd+mobilenet_v1, to detect objects in images. It loads a TFLite model, reads an image file, preprocesses it (resizes the image based on the input_details and output details of the model, and converts to RGB format), runs inference, maps the highest probability output index to a class name using a label file, and stores the results in a text file."
        # self.model_path = "models/ssd-mobilenet_v1/detect.tflite"
        # self.label_path = "models/ssd-mobilenet_v1/labelmap.txt"
        # self.input_description = "Read a single image file from the given input_path"
        # self.input_path = "data/object_detection/horse_sheep.jpg"
        # self.output_description = "write the results in a text file in the given output_path. the format is for reach detection, put it as a row in the text file, with the following five-columned format: <class_name> <confidence> <left> <top> <right> <bottom>"
        # self.output_path = (
        #     "results/object_detection/test_results/horse_sheep_detections.txt"
        # )
        self.target_device = "Raspberry Pi 4B"
        self.application_name = "Object Detection via a video file"
        self.application_description = "This application uses a TFLite model ssd-mobilenet_v1, to detect objects in the mediafile. It loads a TFLite model, reads an video file, preprocesses it (resizes the video frame based on the input_details and output details of the model, and converts to RGB format), runs inference, maps the highest probability output index to a class name using a label file, and draw the boxes with texts of labels and calculated mAP(mean average precision) on the video frame, and stores the video file to the given output_path."
        self.model_path = "models/ssd-mobilenet_v1/detect.tflite"
        self.label_path = "models/ssd-mobilenet_v1/labelmap.txt"
        self.input_description = "Read a single video file from the given input_path"
        self.input_path = "data/object_detection/sheeps.mp4"
        self.output_description = "Output the video file with rectangles drew on the detected objects, along with texts of labels and calculated mAP(mean average precision)"
        self.output_path = "results/object_detection/test_results/sheeps_detections.mp4"

        if any(
            not attr
            for attr in [
                self.application_description,
                self.model_path,
                self.input_description,
                self.output_description,
            ]
        ):
            raise ValueError(
                "Application description, model path, input description, and output description must be set!"
            )

        self.logger.info(
            f"{self.get_session_id()}User input acquired for task: '{self.application_name}'"
        )

    def get_prompt_template(self, template_type: str) -> list[PromptTemplate | str]:
        """Loads the appropriate prompt template from TMPL_PSG."""
        prompts_list: list[PromptTemplate | str] = [TMPL_PSG.context_pro_tem]   

        if template_type == "task_generate_pysketch":
            system_prompt_format = TMPL_PSG.format_code_gen_pro_tem
            task_prompt = PromptTemplate.from_template(TMPL_PSG.code_gen_pro_tem)
        elif template_type == "error_handling_code_gen":
            system_prompt_format = TMPL_PSG.format_code_gen_pro_tem
            task_prompt = PromptTemplate.from_template(
                TMPL_PSG.error_handling_code_gen_pro_tem
            )
        else:
            # Use the current class name in the error message
            raise ValueError(
                f"Unknown template type for PySketchGenerator: {template_type}"
            )

        prompts_list.append(system_prompt_format)
        prompts_list.append(task_prompt)
        return prompts_list

    # def _get_formatted_core_logic(self) -> str:
    #     """Formats the core logic reference template with runtime values."""
    #     return TMPL_PSG.core_logic_reference_tem.format(
    #         input_source=self.input_source,
    #         output_method=self.output_method,
    #         output_file_path=self.output_file_path,
    #         # Add other formatting if needed
    #     )

    def compose_pysketch_generation_prompt(self):
        """Composes the prompt for generating the Python sketch."""
        prompts = self.get_prompt_template("task_generate_pysketch")
        
        static_core_logic = TMPL_PSG.core_logic_reference_tem

        # Format the final task prompt with specific details
        prompts[2] = prompts[2].format(
            application_name=self.application_name,
            application_description=self.application_description,
            model_path=self.model_path,
            label_path=self.label_path,  # Pass optional label path
            input_description=self.input_description,
            input_path=self.input_path,
            output_description=self.output_description,
            output_path=self.output_path,
            core_logic_reference_formatted=static_core_logic,
            confidence_threshold=self.confidence_threshold,
            target_device=self.target_device,
        )
        return prompts

    def compose_error_handling_prompt(self, code: str, error: str):
        """Composes the prompt to ask the LLM to fix the code based on an error."""
        prompts = self.get_prompt_template("error_handling_code_gen")
        
        static_core_logic = TMPL_PSG.core_logic_reference_tem

        prompts[2] = prompts[2].format(
            application_name=self.application_name,
            faulty_code=code,
            error_message=error,
            model_path=self.model_path,
            label_path=self.label_path,  # Pass optional label path
            input_description=self.input_description,
            input_path=self.input_path,
            output_description=self.output_description,
            output_path=self.output_path,
            core_logic_reference_formatted=static_core_logic,
            confidence_threshold=self.confidence_threshold,
            target_device=self.target_device,
        )
        return prompts


    def do_pysketch_generation(self, max_retries: int = 5) -> tuple[bool, str, str]:
        """
        Handles the Python sketch generation process with retry logic.

        This method attempts to generate a valid Python sketch by invoking the LLM,
        extracting code from its response, and validating the generated code.
        If validation fails, it retries the process up to max_retries times.

        Args:
            max_retries: Maximum number of attempts to generate a valid sketch

        Returns:
            tuple[bool, str, str]: (is_valid, generated_code, error_message)
                - is_valid: Whether a valid sketch was generated
                - generated_code: The generated Python code
                - error_message: Error message if generation failed
        """
        generated_code = ""
        is_valid = False
        error_message = ""
        generation_name_prefix = (
            f"{self.trace_id[:4]}_{self.get_task_name(short=True)}_gen"
        )
        for attempt in range(max_retries):
            try:
                self.logger.info(
                    self.get_session_id() + f"Attempt {attempt + 1} of {max_retries}..."
                )

                # Prepare prompt based on attempt number
                if attempt == 0:
                    # First attempt: Generate initial sketch
                    generation_name = f"{generation_name_prefix}_attempt#1"
                    prompt_to_run = self.compose_pysketch_generation_prompt()
                else:
                    # Subsequent attempts: Fix previous error
                    generation_name = f"{generation_name_prefix}_attempt#{attempt+1}"
                    prompt_to_run = self.compose_error_handling_prompt(
                        generated_code if generated_code is not None else "None", error_message
                    )

                # Invoke LLM
                response_raw = self.invoke_llm_with_trace(
                    prompts=prompt_to_run,
                    generation_name=generation_name,
                )

                # Extract code from response
                generated_code = self.extract_code(response_raw, language="python")

                if generated_code and generated_code.strip():
                
                    generated_code = generated_code.replace(
                        "ai-edge-litert", "ai_edge_litert"
                    )
                    
                    generated_code = self.check_placeholders(generated_code, {"model_path":self.model_path, "input_path":self.input_path, "output_path":self.output_path, "label_path":self.label_path,"model_name":self.model_name, "confidence_threshold":self.confidence_threshold})

                    root_dir = Path(os.path.dirname(__file__)).parents[1]
                    validation_workspace = os.path.join(
                        root_dir, "results", "object_detection", "sketches"
                    )
                    os.makedirs(validation_workspace, exist_ok=True)
                    error_message = self.execute_code(
                        generated_code, validation_workspace, is_arduino=False
                    )

                    if error_message is None:
                        self.logger.info(
                            self.get_session_id()
                            + f"Successfully generated valid Python sketch on attempt {attempt + 1}."
                        )
                        return True, generated_code, ""
                else:
                    generated_code = generated_code if generated_code else "None"
                    error_message = "No valid Python code block found in LLM response."
                    self.logger.error(
                        self.get_session_id()
                        + error_message
                    )
                    
 
                if attempt == max_retries - 1:
                 
                    return False, generated_code, f"Max retries reached with failure. Last error from execution: {str(error_message)}"
                
                # If validation failed
                self.logger.error(
                    self.get_session_id()
                    + f"Attempt {attempt + 1} failed validation: {str(error_message)}"
                )
            except Exception as e:
                generated_code = generated_code if generated_code else "None"
                # Check for CUDA errors first (highest priority)
                if "CUDA error" in str(e):
                    error_message = f"CUDA error detected: {e}"
                    return False, generated_code, error_message
                    
                elif "APIConnectionError" in str(e):
                    error_message = f"Ollama error: {e}"
                    return False, generated_code, error_message
                
                if attempt == max_retries - 1:
                    return False, generated_code, f"Max retries reached with failure. Last error from execution: {str(error_message)}"
                
                
                error_message = f"Unexpected exception during generation: {e}\nTraceback: {traceback.format_exc()}"
                self.logger.error(
                    self.get_session_id()
                    + error_message
                )

   


    def run(self, max_retries: int = 5) -> None:
        """
        Orchestrates the Python sketch generation process.

        This method handles the overall flow of the sketch generation process,
        including user input acquisition, sketch generation, and error handling.

        Args:
            max_retries: Maximum number of generation attempts

        Raises:
            KeyboardInterrupt: If the process is interrupted
            APIConnectionError: If an API connection error occurs
            Exception: For any other unexpected errors
        """
        self.logger.info(
            self.get_session_id() + f"Starting {self.__class__.__name__} run..."
        )
        try:
            # Get user input for sketch generation
            self.get_user_input()

            # Generate Python sketch with retry logic
            is_valid, generated_code, error_message = self.do_pysketch_generation(
                max_retries
            )

            if is_valid:
                # Handle successful generation
                self.generated_sketch = generated_code
 
                self.trace.update(
                    output={"status": "success", "sketch": self.generated_sketch}
                )
                self.logger.info(
                    self.get_session_id()
                    + f"{self.__class__.__name__} run finished successfully."
                )
            else:
                # Handle failed generation
                self.logger.error(
                    self.get_session_id()
                    + f"Failed to generate valid Python sketch.  {str(error_message)}"
                )
                self.raise_error_tracing(
                    RuntimeError(f"Failed. Last error: {str(error_message)}")
                )
                self.trace.update(
                    output={
                        "status": "failed",
                        "last_code": generated_code,
                        "last_error": error_message,
                    }
                )
                self.logger.error(
                    self.get_session_id()
                    + f"{self.__class__.__name__} run finished with failure."
                )

        except KeyboardInterrupt as e:
            # Handle keyboard interrupts
            e.args = ("Keyboard interrupt received",) + e.args
            self.logger.error(self.get_session_id() + str(e))
            self.trace.update(
                output={
                    "status": "failed",
                    "last_error": "Keyboard interrupt received. " + str(e),
                }
            )
            self.raise_error_tracing(e)
            self.trace.update(output={"status": "failed", "last_error": str(e)})
            raise  # Re-raise after handling

        except Exception as e:
            # Handle unexpected errors
            self.logger.error(
                self.get_session_id()
                + f"An unexpected error occurred: {traceback.format_exc()}"
            )
            self.trace.update(
                output={
                    "status": "failed",
                    "last_error": "An unexpected error occurred. " + str(e),
                }
            )
            self.raise_error_tracing(e)
            self.trace.update(output={"status": "failed", "last_error": str(e)})


if __name__ == "__main__":

    # Load environment variables from .env file for API keys, etc.
    load_dotenv()
    
    # --- Configuration for direct test run ---
    # Select the LLM strategy (ensure corresponding env vars are set in .env)
    # Example: Use Ollama with codestral
    # test_llm_type = "openai"
    # test_model_name = "gpt-4o"
    # Example: Use OpenAI
    # test_llm_type = "openai"
    # test_model_name = "gpt-4o"
    test_llm_type = "ollama"
    test_model_name = "codestral:latest"

    # Generate a unique trace ID for this test run
    test_trace_id = test_model_name + "_" + str(uuid.uuid4()).split("-")[0]
    print(f"Using Trace ID: {test_trace_id}")
    # --- End Configuration ---

    # Create LLM Strategy
    try:
        llm_strategy = LLMFactory.create_llm(test_llm_type, model_name=test_model_name, parameters=True)
    except ValueError as e:
        print(f"Error creating LLM strategy: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during LLM setup: {e}")
        sys.exit(1)

    # Instantiate the processor
    # Not running as part of a benchmark batch, so benchmark=False, num_run/session_id=None
    processor = PySketchGenerator(
        llm_strategy=llm_strategy,
        trace_id=test_trace_id,
        benchmark=False,
        num_run=None,
        session_id=None,
        self_host_langfuse_flag=True,
    )
    print(f"Running {__file__} directly for a single test...")
    # Run the processor's main logic
    processor.run()

    print(f"Direct test run completed for trace ID: {test_trace_id}")
