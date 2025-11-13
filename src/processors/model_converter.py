import os
import sys
import traceback
import uuid

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langfuse import Langfuse

# Add this line to include the directory containing prompt_templates
import prompt_templates.TMPL_MC as TMPL_MC
from base.base_processor import BaseProcessor
from base.llm_strategy import (
    LLMStrategy,
    OllamaLiteLLMStrategy,
    OpenAILiteLLMStrategy,
    OpenAIStrategy,
)


class ModelConverter(BaseProcessor):
    """
    ModelConverter is responsible for converting machine learning models to a specified format.
    It generates the necessary conversion code using an LLM, executes the code, and handles any errors.
    """

    load_dotenv()

    def __init__(
        self, llm_strategy, trace_id, num_run=None, benchmark=False, session_id=None, self_host_langfuse_flag=False
    ):
        """
        Initializes the ModelConverter with the given parameters.

        Args:
            llm_strategy: The language model strategy object providing invoke() method.
            trace_id: Unique identifier for tracing the whole run.
            num_run: Optional run number, to locate a run in a batch run.
            benchmark: Flag to indicate benchmarking(batch run).
            session_id: Optional batch identifier, to locate a batch test among different times of batch tests.
        """
        super().__init__(
            llm_strategy,
            trace_id=trace_id,
            num_run=num_run,
            benchmark=benchmark,
            session_id=session_id,
            task_name=self.get_task_name(),
            self_host_langfuse_flag=self_host_langfuse_flag,
        )
        self.ptq = True  # Post-training quantization flag
        self.converted_model_path = ""
        self.original_model_path = ""
        self.quant_req_e = ""  # example: "with 8-bit integer quantization"
 

    def get_user_input(self):
        """
        Acquires and sets the user input parameters required for model conversion.
        """
        self.original_model_path = "models/fruit_to_emoji/og_model/model.keras"
        self.dataset_path = "data/fruit_to_emoji/SampleData/fruit_data.csv"
        self.input_datatype = "float32"

        if self.ptq:
            self.output_datatype = "uint8"
            self.quant_req_e = " with 8-bit integer quantization"
            self.converted_model_path = (
                "models/fruit_to_emoji/tflite_model/model_quant.tflite"
            )
        else:
            self.output_datatype = "float32"
            self.quant_req_e = " without quantization"
            self.converted_model_path = (
                "models/fruit_to_emoji/tflite_model/model.tflite"
            )
        self.logger.info(
            self.get_session_id()
            + f"User input acquired, PTQ: {self.ptq}, input_datatype: {self.input_datatype}, output_datatype: {self.output_datatype}"
        )

    def get_prompt_template(self, template_type) -> list[str|PromptTemplate]:
        """
        Retrieves the appropriate prompt template based on the template type.

        Args:
            template_type: The type of template to retrieve.

        Returns:
            A PromptTemplate object or None if not found.
        """
        if template_type == "error_handling":
            task_prompt = PromptTemplate.from_template(
                TMPL_MC.error_handling_code_gen_pro_tem
            )
            system_prompt_format = TMPL_MC.format_pro_tem
        elif template_type == "task":
            task_prompt = PromptTemplate.from_template(TMPL_MC.code_gen_pro_tem)
            system_prompt_format = TMPL_MC.format_pro_tem
        else:
            raise ValueError(f"Unknown template type: {template_type}")
        
        prompts_list = [
            TMPL_MC.context_pro_tem,
            system_prompt_format,
            task_prompt,
        ]
 
        return prompts_list

    def do_conversion_code_generation(self, max_retries: int = 5) -> tuple[bool, str|None, str|None]:
        """
        Generates and executes the model conversion code using an LLM.
        Retries upon failure up to max_retries times.

        Args:
            max_retries: Maximum number of retries for code generation.

        Raises:
            Exception: If code generation fails after maximum retries.
        """
        code = None
        error = None

        for attempt in range(1, max_retries + 1):
            try:
                if attempt == 1:

                    prompts = self.get_prompt_template("task")

                    prompts[2] = prompts[2].format(
                        dataset_path=self.dataset_path,
                        original_model_path=self.original_model_path,
                        converted_model_path=self.converted_model_path,
                        quant_req_n=self.quant_req_e.replace(" ", "_"),
                        quant_req_e=self.quant_req_e,
                        input_datatype=self.input_datatype,
                        output_datatype=self.output_datatype,
                    )
                    self.logger.info(
                        self.get_session_id() + "Generating code for the first time..."
                    )
                else:
                    self.logger.info(
                        self.get_session_id()
                        + f"Trying to solve the error... Re-attempt {attempt}..."
                    )
                    prompts = self.get_prompt_template("error_handling")
                    if not prompts:
                        raise ValueError("Error handling template not found")

                    prompts[2] = prompts[2].format(
                        executed_code=str(code),
                        error_info=str(error),
                        quant_req_e=self.quant_req_e,
                        original_model_path=self.original_model_path,
                        converted_model_path=self.converted_model_path,
                        input_datatype=self.input_datatype,
                        output_datatype=self.output_datatype,
                    )

                response_raw = self.invoke_llm_with_trace(
                    prompts,
                    generation_name=f"{self.trace_id[:2]}_{self.get_task_name(short=True)}_"
                    f"{'conversion_code_gen' if attempt == 1 else 'error_handling'}"
                    f"{'_attempt#' + str(attempt) if attempt > 1 else ''}",
                )
                generated_code = self.extract_code(response_raw,language="python")
                
                # if the number of lines of the generated_code is less than 20, raise and error
                if not generated_code or len(generated_code.splitlines()) < 20:
                    error_message = (
                        "Generated code is not complete. Please ensure to generate a complete Python script."
                    )
                    self.logger.error(
                        self.get_session_id() + error_message
                    )
                    raise OSError(error_message)
          
                
                if not generated_code.strip() or not generated_code:
                    self.logger.error(
                        self.get_session_id()
                        + "No Python code block found in LLM response."
                    )
                    error_message = "No Python code block found in LLM response."
                    if attempt == max_retries - 1:
                        return False, generated_code, error_message
                    continue  # Go to next retry, hoping LLM formats correctly
                self.logger.info(
                    self.get_session_id()
                    + f"{'First attempt' if attempt == 1 else 'Error handling attempt NO.' + str(attempt)} code received. Executing..."
                )
                code = generated_code
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                # ! [x] BUG: This execute_code invocation is outdated.
                error = self.execute_code(
                    code,
                    os.path.join(
                        "models/fruit_to_emoji/tflite_model",
                        "tmp_converter" + timestamp + ".py",
                    ),
                    is_arduino=False,
                )
                if not error:
                    self.logger.info(
                        self.get_session_id() + "Code execution successful."
                    )
                    return True, code, ""
                else:
                    self.logger.error(
                        self.get_session_id()
                        + f"Error in code execution attempt NO.{attempt}: {error}"
                    )

            except Exception as e:
                error = str(e)
                if "CUDA error" in error:

                    return False, None, f"[FATAL]: CUDA error occurred.\nError: {str(e)}\nTraceback: {traceback.format_exc()}"

                elif "APIConnectionError" in error:
                    
                    return False, None, f"[FATAL]: Ollama error occurred.\nError: {str(e)}\nTraceback: {traceback.format_exc()}"
                else:
                    self.logger.error(
                        self.get_session_id()
                        + f"Exception caught in prompting attempt {attempt}, loop continues.\nError: {str(e)}\nTraceback: {traceback.format_exc()}"
                    )
        self.logger.error(
            self.get_session_id()
            + f"Failed to generate valid code after the max {max_retries} attempts. Last error from code execution: {error}"
        )
        # self.raise_error_tracing(
        #     Exception(
        #         f"Failed to generate valid code after the max {max_retries} attempts, programme terminated."
        #     )
        # )
        return False, code, f"Failed to generate valid code after the max {max_retries} attempts. Last error from code execution: {error}"

    def run(self):
        """
        Executes the model conversion process by acquiring user input and initiating code generation.
        """
        self.trace.span(
            name=f"start_{self.get_task_name()}",
        )
        print(
            "Model name: " + self.llm_strategy.model_name,
            "; Trace name: " + self.trace_name,
            "; Batch ID: " + self.session_id if self.session_id else "",
        )
        self.logger.info(self.get_session_id() + "Model conversion process started.")

        try:
            self.get_user_input()

            status, generated_code, error_message = self.do_conversion_code_generation(max_retries=5)
            if status:
                self.logger.info(
                    self.get_session_id() + "Model conversion process completed."
                )
                self.trace.update(
                        output={"status": "success", "code": generated_code}
                    )
            else:
                self.logger.error(
                        self.get_session_id()
                        + f"Failed to generate valid Python code.  {str(error_message)}"
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

# Temporary test client to run the ModelConverter independently
if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    from factories.llm_factory import LLMFactory

    load_dotenv()

    # Choose the strategy type here: "langchain_openai" or "direct_openai"
    # llm_strategy = LLMFactory.create_llm(
    #     "openai", api_key=os.getenv("OPENAI_API_KEY"), model_name="gpt-4o-mini"
    # )

    model_name = "gemini-1.5-flash"

    # llm_strategy = LLMFactory.create_llm("ollama", model_name=model_name)
    llm_strategy = LLMFactory.create_llm("gemini", model_name=model_name)
    # trace_id is used to stack the subtasks of a run together in LangFuse
    trace_id = str(uuid.uuid4()).split("-")[0]

    model_converter = ModelConverter(
        llm_strategy,
        trace_id=trace_id,
        benchmark=False,
    )
    model_converter.run()
