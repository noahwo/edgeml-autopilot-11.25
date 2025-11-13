"""!
DataProcessor Module

This module defines the DataProcessor class, responsible for processing datasets
including generating data processing code, applying operations, and handling errors.
It leverages language models to assist in generating accurate and executable data processing scripts.
"""

import os
import sys
import traceback
import uuid

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json
import shutil
from datetime import datetime
from typing import Dict, List, Union

# import pandas as pd
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

import prompt_templates.TMPL_DP as data_proc_template
from base.base_processor import BaseProcessor

# from base.llm_strategy import (
#     LLMStrategy,
#     OllamaLiteLLMStrategy,
#     OpenAILiteLLMStrategy,
#     OpenAIStrategy,
# )


# from langfuse import Langfuse

# from langsmith import traceable, trace


class DataProcessor(BaseProcessor):
    """
    DataProcessor is responsible for processing datasets by generating and executing
    data processing code as per suggested operations. It interacts with an LLM to
    generate code for each data processing operation and applies it to the dataset.
    """

    load_dotenv()

    def __init__(
        self, llm_strategy, trace_id, num_run=None, benchmark=False, session_id=None, self_host_langfuse_flag=False
    ):
        """
        Initializes the DataProcessor with the given parameters.

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
        self.current_operation = {}
        self.list_processing_already_applied = []
        self.purpose = ""
        self.dataset_intro = ""
        self.suggestion_table = {}
        self.current_op_count = 0
        self.total_op_count = 0
        self.files_in_playground_before_execution = []

        self.playground_dir = "./data/fruit_to_emoji/playground/"


    def get_user_input(self):
        """
        Acquires and sets the user input parameters required for data processing.
        Prepares the playground directory by copying necessary data files.
        """
        source_dir = "./data/fruit_to_emoji/SampleData/"

        # Remove all .csv files under "playground/"
        for file in os.listdir(self.playground_dir):
            if file.endswith(".csv"):
                file_path = os.path.join(self.playground_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        # Copy all .csv files from "./data/fruit_to_emoji/SampleData/" to "./data/fruit_to_emoji/playground/"

        shutil.copy(os.path.join(source_dir, "fruit_data.csv"), self.playground_dir)

        self.purpose = "classify fruits among apple, orange, and banana"

        self.dataset_path = "./data/fruit_to_emoji/playground/fruit_data.csv"
        self.dataset_intro = """This dataset contains RGB data for a collection of fruits, categorized into three classes: "apple", "orange", and "banana". The dataset contains four columns, respectively represents the fruit name, and the Red Data, Green Data, and Blue data of the fruit."""

        self.logger.info(
            self.get_session_id() + "User input acquired.",
        )

    # @lru_cache(maxsize=32)
    def get_prompt_template(self, template_type) -> List[str| PromptTemplate]:

        # task_prompt is PromptTemplate, system_prompt_format is string
        if template_type == "suggestion_gen":
            task_prompt = PromptTemplate.from_template(
                data_proc_template.suggestion_gen_pro_tem
            )
            system_prompt_format = data_proc_template.format_suggestion_pro_tem
        elif template_type == "error_handling_suggestion":
            task_prompt = PromptTemplate.from_template(
                data_proc_template.error_handling_suggestion_pro_tem
            )
            system_prompt_format = data_proc_template.format_suggestion_pro_tem
        elif template_type == "code_gen":
            task_prompt = PromptTemplate.from_template(
                data_proc_template.code_gen_pro_tem
            )
            system_prompt_format = data_proc_template.format_code_gen_pro_tem
        elif template_type == "error_handling_code_gen":
            task_prompt = PromptTemplate.from_template(
                data_proc_template.error_handling_code_gen_pro_tem
            )
            system_prompt_format = data_proc_template.format_code_gen_pro_tem
        else:
            raise ValueError(f"Unknown template type: {template_type}")
            return None

        prompts_list = [
            data_proc_template.context_pro_tem,
            system_prompt_format,
            task_prompt,
        ]
        return prompts_list
    
    def raise_and_exit_dp(self, error_message:str):
     
        self.logger.error(self.get_session_id()+ error_message)
        self.raise_error_tracing( Exception(error_message) )
        self.trace.update(output={"status": "failed", "last_error": error_message})
        sys.exit(1)


    def do_suggestion_table_generation(self, max_retries):
        """
        Generates a suggestion table for data processing operations using the LLM.
        Retries with error handling if JSON parsing fails.

        Args:
            max_retries: Maximum number of retry attempts

        Returns:
            A dictionary containing suggested data processing operations.
        """
        self.dataset_summary()
        response_raw = None
        error = None

        for attempt in range(1, max_retries + 1):
            try:
                if attempt == 1:
                    prompt_to_run = self.get_prompt_template("suggestion_gen")

                    prompt_to_run[2] = prompt_to_run[2].format(
                        purpose=self.purpose,
                        dataset_summary=self.dataset_summary_str,
                        dataset_intro=self.dataset_intro,
                    )
                    self.logger.info(
                        self.get_session_id()
                        + "Generating suggestion table using LLM attempt #1...",
                    )
                else:
                    self.logger.info(
                        self.get_session_id()
                        + f"Retrying suggestion table generation (attempt #{attempt})..."
                    )
                    prompt_to_run = self.get_prompt_template(
                        "error_handling_suggestion"
                    )
                    prompt_to_run[2] = prompt_to_run[2].format(
                        purpose=self.purpose,
                        dataset_summary=self.dataset_summary_str,
                        dataset_intro=self.dataset_intro,
                        executed_code=response_raw,
                        error_info="Failed to parse response as valid JSON: " + error if error else "",
                    )

                response_raw = self.invoke_llm_with_trace(
                    prompt_to_run,
                    generation_name=self.trace_id[:2]
                    + "_"
                    + self.get_task_name(short=True)
                    + f"_suggestion_table_gen_attempt#{attempt}",
                )

                content = (
                    response_raw.strip()
                    .replace("```json", "")
                    .replace("```", "")
                    .replace("[", "")
                    .replace("]", "")
                    .replace("\\n'", "")
                    .strip()
                )

                self.suggestion_table = self.extract_suggestion_table(content)

                self.total_op_count = len(self.suggestion_table)

                self.logger.info(
                    self.get_session_id()
                    + "Suggestions received and parsed successfully:",
                )
                self.logger.info(
                    f"Suggestions: {json.dumps(self.suggestion_table, indent=2)}",
                )
                return True, self.suggestion_table, ""

            except json.JSONDecodeError as e:
                self.logger.error(
                    self.get_session_id()
                    + f"Attempt {attempt}: Failed to parse LLM response as a dictionary: {content}"
                )
                error = str(e)
                if attempt == max_retries:
                    error_message=f"Failed to generate valid suggestion table after {max_retries} attempts. Last error: {error}"
                    self.raise_and_exit_dp(error_message)
            except Exception as e:
                error = str(e)
                if "CUDA error" in error:
                    error_message=f"[FATAL]: CUDA error detected: {e}"
                    self.raise_and_exit_dp(error_message)
                elif "APIConnectionError" in error:
                    error_message= f"[FATAL]: Ollama error: {e}"
                    self.raise_and_exit_dp(error_message)
                else:
                    self.logger.error(
                        self.get_session_id()
                        + f"Exception caught in prompting attempt {attempt}, loop continues.\nError: {str(e)}\nTraceback: {traceback.format_exc()}"
                    )

    def do_operation_code_generation(self, max_retries=5):
        """
        Generates and executes the code for each data processing operation using the LLM.
        Retries upon failure up to max_retries times.

        Args:
            max_retries: Maximum number of retries for code generation.

        Raises:
            Exception: If code generation fails after maximum retries.
        """
  
        code = None
        error = None
        current_op = self.current_operation

        for attempt in range(1, max_retries + 1):

            if attempt == 1:

                prompts = self.get_prompt_template("code_gen")
                prompts[2] = prompts[2].format(
                    operation_n_explanation=current_op["explanation"],
                    current_operation_name=current_op["operation"],
                    # current_operation=f"{current_op['operation']}: {current_op['explanation']}",
                    dataset_path=self.dataset_path,
                    list_processing_already_applied=str(
                        self.list_processing_already_applied
                    ),
                    dataset_summary=self.dataset_summary_str,
                )
            else:
                self.logger.info(
                    self.get_session_id()
                    + f"Trying to solve the error, re-attempting... (attempt NO.{attempt})",
                )
                prompts = self.get_prompt_template("error_handling_code_gen")

                prompts[2] = prompts[2].format(
                    current_operation=f"{current_op['operation']}: {current_op['explanation']}",
                    executed_code=code,
                    error_info=error,
                    dataset_path=self.dataset_path,
                    list_processing_already_applied=self.list_processing_already_applied,
                    dataset_summary=self.dataset_summary_str,
                )

            response_raw = self.invoke_llm_with_trace(
                prompts,
                generation_name=f"{self.trace_id[:2]}_{self.get_task_name(short=True)}_"
                f"{'operation_code_gen' if attempt == 1 else 'error_handling'}"
                f"#{len(self.list_processing_already_applied) + 1}"
                f"{'_attempt#' + str(attempt) if attempt > 1 else ''}",
            )
            try:
                code = self.extract_code(response_raw)

                self.logger.info(
                    self.get_session_id()
                    + f"Operation #{self.current_op_count}/{self.total_op_count} "
                    + current_op["operation"]
                    + ": "
                    + f"{'First attempt' if attempt == 1 else 'Error handling, attempt NO.' + str(attempt)} code received. Executing...",
                )
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                error = self.execute_code(
                    code,
                    self.playground_dir + "tmp_" + timestamp + ".py",
                )
                if not error:
                    self.update_dataset_path(response_raw)
                    self.logger.info(
                        self.get_session_id() + "Code execution successful."
                    )
                    return None
                else:
                    self.logger.error(
                        self.get_session_id()
                        + f"Error in code execution attempt NO.{attempt} for operation  #{self.current_op_count}/{self.total_op_count} "
                        + str(current_op)
                        + ": "
                        + error
                    )
                if attempt == max_retries - 1:
                    error_message = f"Failed to generate valid code after the max {max_retries} attempts. Last error from code execution: {error}"  
                    self.raise_and_exit_dp(error_message)
                    
            except Exception as e:
                error = str(e)
                if "CUDA error" in error:
                    error_message=f"[FATAL]: CUDA error detected: {e}"
                    self.raise_and_exit_dp(error_message)
                elif "APIConnectionError" in error:
                    error_message= f"[FATAL]: Ollama error: {e}"
                    self.raise_and_exit_dp(error_message)
               
                elif attempt == max_retries - 1:
                    error_message=f"Failed to generate valid code after the max {max_retries} attempts. Last error from code execution: {error}"
                    self.raise_and_exit_dp(error_message)
                else:
                    self.logger.error(
                        self.get_session_id()
                        + f"Exception caught in prompting attempt {attempt}: {str(e)}, loop continues.\nError: {str(e)}\nTraceback: {traceback.format_exc()}"
                    )

    def update_dataset_path(self, content):
        """
        Extracts dataset paths from LLM response in various formats.
        Recursively searches through nested dictionaries to find all .csv paths.

        Args:
            content: The raw response from the LLM, can be string or object

        Returns:
            None

        Raises:
            ValueError: If no valid dataset paths are found
        """

        # Remove the first block of ```python ... ``` from content
        content = content.split("```python", 1)[-1]

        # Extract content from possible code block types
        for block_type in ["```json", "```python", "```"]:
            if block_type in content:
                content = content.split(block_type, 1)[-1].split("```", 1)[0].strip()
                break

        # Set the extracted content to self.dataset_path
        self.dataset_path = content

    def extract_suggestion_table(self, content):
        """
        Extract suggestion table from LLM output content.
        Handle both single dictionary and multiple dictionaries formats.

        Args:
            content (str): LLM output content

        Returns:
            dict: Combined suggestion table

        Raises:
            json.JSONDecodeError: If content cannot be parsed as JSON
        """
        # Remove any whitespace and newlines from start/end
        content = content.strip()

        # First try to parse as a single dictionary
        try:
            suggestion_table = json.loads(content)
            return suggestion_table
        except json.JSONDecodeError:
            # If that fails, try parsing as multiple dictionaries
            suggestion_table = {}
            # Split by newlines and process each line
            for line in content.split("\n"):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        # Parse each line as a dictionary
                        line_dict = json.loads(line)
                        # Merge into main dictionary
                        suggestion_table.update(line_dict)
                    except json.JSONDecodeError:
                        continue

            if suggestion_table:  # If we successfully parsed any dictionaries
                return suggestion_table

            # If we get here, we couldn't parse the content - raise JSONDecodeError
            # instead of ValueError to match error handling in do_suggestion_table_generation
            raise json.JSONDecodeError(
                "Could not parse suggestion table from content", content, 0
            )

    def run(self):
        """
        Executes the data processing workflow by acquiring user input, generating a suggestion table, and performing each suggested data processing operation.
        """
        print(
            "Model name: " + self.llm_strategy.model_name,
            "; Trace name: " + self.trace_name,
            "; Batch ID: " + self.session_id if self.session_id else "",
        )

        self.logger.info(
                self.get_session_id() + "Data processing started.",
            )
        try:
 
            self.get_user_input()
            self.do_suggestion_table_generation(max_retries=int(5))

            self.current_op_count = 1
            for operation_n, operation_n_explanation in self.suggestion_table.items():

                self.current_operation = {
                    "operation": operation_n,
                    "explanation": operation_n_explanation,
                }
                self.logger.info(
                    self.get_session_id()
                    + f"STARTED: Operation #{self.current_op_count}/{self.total_op_count} "
                    + operation_n
                    + ".",
                )
                self.do_operation_code_generation(max_retries=int(5))

                self.list_processing_already_applied.append(
                    {str(operation_n): str(operation_n_explanation)}
                )
                self.logger.info(
                    self.get_session_id()
                    + f"COMPLETED: Operation #{self.current_op_count}/{self.total_op_count} "
                    + operation_n
                    + ".",
                )
                self.logger.info(self.get_session_id() + 40 * "#")
                self.current_op_count += 1
            

                name=f"end_{self.get_task_name()}",
                status_message="Data processing completed successfully.",
            
            self.trace.update(
                    output={"status": "success"}
                )
            self.logger.info(
                self.get_session_id()
                + f"Data processing completed, here are the executed operations:  {json.dumps(self.suggestion_table, indent=2).strip()[1:-1]}",
            )
        except KeyboardInterrupt as e:
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
            raise 
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


# Temporary test client to run the DataProcessor independently
if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    from factories.llm_factory import LLMFactory

    load_dotenv()

    # openai_api_key = os.getenv("OPENAI_API_KEY")
    # model_name = "gemini-1.5-flash"
    model_name = "gpt-4o"
    # llm_strategy = LLMFactory.create_llm("ollama", model_name=model_name)
    llm_strategy = LLMFactory.create_llm("openai", model_name=model_name)

    # llm_strategy = LLMFactory.create_llm("openai", api_key=openai_api_key, model_name=model_name)

    # trace_id is used to stack the subtasks of a run together in LangSmith. 8-bit
    trace_id = str(uuid.uuid4()).split("-")[0]

    data_processor = DataProcessor(
        llm_strategy,
        benchmark=False,
        trace_id=trace_id,
        # session_id="pseudo_session_id",
    )
    data_processor.run()
    # data_processor.get_user_input()
    # data_processor.logger.info(
    #     data_processor.get_session_id() + "Data processing started.",
    # )
