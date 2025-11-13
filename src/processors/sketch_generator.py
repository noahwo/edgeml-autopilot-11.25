import json
import os
import sys
import traceback
from typing import List, Union
import uuid

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


from langchain_core.prompts import PromptTemplate

import prompt_templates.TMPL_SG as TMPL_SG
from base.base_processor import BaseProcessor


class SketchGenerator(BaseProcessor):
    """!
    SketchGenerator is responsible for generating Arduino .ino sketches based on
    provided application specifications and programming guidelines and compiling
    Arduino sketches (.ino files).
    """

    def __init__(
        self, llm_strategy, trace_id, num_run=None, benchmark=False, session_id=None, self_host_langfuse_flag=False
    ):
        """
        Initializes the SketchGenerator with the given parameters.

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
        self.application_name = ""
        self.application_description = ""
        self.board_fullname = ""
        self.classification_classes = ""
        self.app_specifications_empty = ""
        self.app_specifications_filled = ""
        self.current_operation = {
            "operation": "write_ino_sketch_for_the_application",
            "explanation": "Generate the .ino sketch code for the application running on the board.",
        }
        self.app_specifications_with_programming_guidelines = ""

    # def get_task_name(self, short=False):
    #     """!
    #     Returns the task name.

    #     Args:
    #         short: If True, return abbreviated task name.

    #     Returns:
    #         Task name as a string.
    #     """
    #     if short:
    #         return "sg"
    #     else:
    #         return "sketch_generator"

    def get_user_input(self):
        """!
        Acquires and sets the user input parameters required for sketch generation.
        """
        self.application_name = "Object Classifier by Color"
        self.application_description = "Uses RGB color sensor input to Neural Network to classify objects and outputs object class to serial using Unicode emojis."
        self.board_fullname = "Arduino Nano 33 BLE Sense"
        self.classification_classes = "Apple, Banana, Orange"
        self.input_datatype = "np.float32"
        self.output_datatype = "np.uint8"
        self.dataset_path = "data/fruit_to_emoji/SampleData/apple.csv"
        self.logger.info(self.get_session_id() + "User input acquired")

    def get_prompt_template(self, template_type: str) -> List[Union[str, PromptTemplate]]:

        if template_type == "error_handling_code_gen":
            system_prompt_format = TMPL_SG.format_code_gen_pro_tem
            task_prompt = PromptTemplate.from_template(
                TMPL_SG.error_handling_code_gen_pro_tem
            )
        elif template_type == "error_handling_spec_filling":
            system_prompt_format = TMPL_SG.format_spec_filling_pro_tem
            task_prompt = PromptTemplate.from_template(
                TMPL_SG.error_handling_spec_filling_pro_tem
            )
        elif template_type == "task_fill_specs":
            system_prompt_format = TMPL_SG.format_spec_filling_pro_tem
            task_prompt = PromptTemplate.from_template(TMPL_SG.spec_filling_pro_tem)
        elif template_type == "task_sketch":
            system_prompt_format = TMPL_SG.format_code_gen_pro_tem
            task_prompt = PromptTemplate.from_template(TMPL_SG.code_gen_pro_tem)
        else:
            raise ValueError(f"Unknown template type: {template_type}")

        prompts_list = [
            TMPL_SG.context_pro_tem,
            system_prompt_format,
            task_prompt,
        ]
        return prompts_list

    def compose_specification_filling_prompt(self) -> List[Union[str, PromptTemplate]]:
        
        
        classification_classes_list = [
            c.strip() for c in self.classification_classes.split(",")
        ]

        # Directly get the template from TMPL_SG
        self.app_specifications_empty = PromptTemplate.from_template(
            TMPL_SG.app_spec_pro_tem
        ).format(
            application_name_hinting_its_purpose=self.application_name,
            application_description=self.application_description,
            board_fullname=self.board_fullname,
            classification_classes=json.dumps(classification_classes_list),
            input_datatype=self.input_datatype,
            output_datatype=self.output_datatype,
            decide_when_generating_code_based_on_given_board_and_application_description="{decide_when_generating_code_based_on_given_board_and_application_description}",
            decide_when_generating_code_based_on_given_data_sample_and_application_description="{decide_when_generating_code_based_on_given_data_sample_and_application_description}",
            programming_guidelines="{programming_guidelines_placeholder_remain_this_untouched}",
        )
        spec_filling_prompts = self.get_prompt_template("task_fill_specs")

        spec_filling_prompts[2] = spec_filling_prompts[2].format(
            app_spec_pro_tem=self.app_specifications_empty,
            dataset_summary=self.dataset_summary_str,
            board_fullname=self.board_fullname,
        )

 
        return spec_filling_prompts

    def compose_spec_filling_error_handling_prompt(self, code: str, error: str) -> List[Union[str, PromptTemplate]]:

        error_handling_spec_filling_prompts = self.get_prompt_template(
            "error_handling_spec_filling"
        )

        error_handling_spec_filling_prompts[2] = error_handling_spec_filling_prompts[
            2
        ].format(
            executed_code=code,
            error_info=error,
            app_spec_pro_tem=self.app_specifications_empty,
            dataset_summary=self.dataset_summary_str,
            board_fullname=self.board_fullname,
        )

        return error_handling_spec_filling_prompts

    def compose_spec_with_guideline_prompt(self):
        
        
        if self.app_specifications_filled is None:
            raise ValueError("app_specifications_filled is None.")

        if self.app_specifications_with_programming_guidelines is None:
            self.app_specifications_with_programming_guidelines = ""

        # ugly situation for the placeholder from returned code may be various formats
        programming_guidelines_prompt = TMPL_SG.programming_guidelines_pro_tem

        if (
            "{programming_guidelines_placeholder_remain_this_untouched}"
            in self.app_specifications_filled
        ):
            self.app_specifications_with_programming_guidelines = (
                self.app_specifications_filled.replace(
                    "{programming_guidelines_placeholder_remain_this_untouched}",
                    programming_guidelines_prompt,
                )
            )
        elif (
            '"{programming_guidelines_placeholder_remain_this_untouched}"'
            in self.app_specifications_filled
        ):
            self.app_specifications_with_programming_guidelines = (
                self.app_specifications_filled.replace(
                    '"{programming_guidelines_placeholder_remain_this_untouched}"',
                    programming_guidelines_prompt,
                )
            )
        elif (
            "programming_guidelines_placeholder_remain_this_untouched"
            in self.app_specifications_filled
        ):
            self.app_specifications_with_programming_guidelines = (
                self.app_specifications_filled.replace(
                    "programming_guidelines_placeholder_remain_this_untouched",
                    programming_guidelines_prompt,
                )
            )
        else:
            # Try to match "guideline": {anytext} pattern
            import re

            def find_matching_brace(s, start):
                count = 1
                i = start
                while count > 0 and i < len(s):
                    if s[i] == "{":
                        count += 1
                    elif s[i] == "}":
                        count -= 1
                    i += 1
                return i if count == 0 else -1

            # Find the programming_guidelines pattern
            match = re.search(r'"programming_guidelines":\s*{', self.app_specifications_filled)
            if match:
                start = match.end() - 1  # Position of opening brace
                end = find_matching_brace(self.app_specifications_filled, start + 1)
                if end != -1:
                    # Replace the entire matched pattern with new guideline
                    old_text = self.app_specifications_filled[match.start() : end]
                    self.app_specifications_with_programming_guidelines = (
                        self.app_specifications_filled.replace(
                            old_text, f'"programming_guidelines": {programming_guidelines_prompt}'
                        )
                    )
                else:
                    raise ValueError(
                        "Found programming guidelines key but couldn't match closing brace"
                    )
            else:
                raise ValueError(
                    "`programming_guidelines_placeholder_remain_this_untouched` not found in the app specifications."
                )

    def compose_sketch_generation_prompt(self, code: str = "", error: str = "") -> List[Union[str, PromptTemplate]]:
        """!
        Composes the prompt for generating the sketch code using the filled specifications.

        Returns:
            A formatted prompt string.

        Raises:
            ValueError: If the programming_guidelines placeholder is not found.
        """
        sketch_generation_prompts = self.get_prompt_template("task_sketch")
        sketch_generation_prompts[2] = sketch_generation_prompts[2].format(
            app_spec_pro_tem=self.app_specifications_with_programming_guidelines,
            dataset_summary=self.dataset_summary_str,
        )

        return sketch_generation_prompts

    def compose_code_gen_error_handling_prompt(self, code, error):
        error_handling_prompts = self.get_prompt_template("error_handling_code_gen")
        if not isinstance(error_handling_prompts[2], PromptTemplate):
            raise TypeError("Expected prompts[2] to be a PromptTemplate object")

        error_handling_prompts[2] = error_handling_prompts[2].format(
            executed_code=str(code),
            error_info=str(error),
            board_fullname=self.board_fullname,
            dataset_summary=self.dataset_summary_str,
            app_spec_pro_tem=str(self.app_specifications_with_programming_guidelines),
        )
        return error_handling_prompts

 
    def do_specification_filling(self, max_retries=5):
        """
        Executes the task to fill in the application specifications using the LLM.
        """
        response = None
        error = None
        generation_name_prefix = f"{self.trace_id[:2]}_{self.get_task_name(short=True)}"
        
        for attempt in range(1, max_retries + 1):
            try:
                if attempt == 1:
                    prompts = self.compose_specification_filling_prompt()
                else:
                    prompts = self.compose_spec_filling_error_handling_prompt(
                        str(response) if response is not None else "", error
                    )

                response = self.invoke_llm_with_trace(
                    prompts,
                    generation_name=f"{generation_name_prefix}_spec_filling_attempt#{attempt}",
                )

                try:
                    app_specifications_filled = response.split("```json")[1].split(
                        "```"
                    )[0]
                    
                    return True, app_specifications_filled, ""
                
                except IndexError:
                    
                    app_specifications_filled = response.split("```")[1].split(
                        "```"
                    )[0]
             
                    return True, app_specifications_filled, ""
                
            except Exception as e:
                error = str(e)
                if "CUDA error" in error:

                    return False, None, f"[FATAL]: CUDA error occurred.\nError: {str(e)}\nTraceback: {traceback.format_exc()}"

                elif "APIConnectionError" in error:
                    
                    return False, None, f"[FATAL]: Ollama error occurred.\nError: {str(e)}\nTraceback: {traceback.format_exc()}"
 
                if attempt == max_retries:
                    return False, None, f"Max attempts reached exceeded. Last error: {str(e)}. "
                
            
                self.logger.error(
                    self.get_session_id()
                    + f"Attempt {attempt + 1} failed validation.\nTraceback: {traceback.format_exc()}. "
                )


    def do_sketch_generation(self, max_retries=5):
        """
        Generates and compiles the sketch code using the LLM. Retries upon failure.

        Args:
            max_retries: Maximum number of retries for code generation.

        Returns:
            True if sketch generation and compilation is successful, False otherwise.

        Raises:
            Exception: If sketch generation fails after maximum retries.
        """
        code = None
        error = None
        generation_name_prefix=f"{self.trace_id[:2]}_{self.get_task_name(short=True)}"
        for attempt in range(1, max_retries + 1):
            try:
                if attempt == 1:
                    prompts = self.compose_sketch_generation_prompt()
                    generation_name=f"{generation_name_prefix}_sketch_code_gen_attempt#1"
                else:
                    self.logger.info(
                        self.get_session_id()
                        + f"Trying to solve the error... Attempt {attempt }..."
                    )
                    generation_name=f"{generation_name_prefix}_sketch_code_gen_attempt#{attempt}"
                    prompts = self.compose_code_gen_error_handling_prompt(code, error)

                response_raw = self.invoke_llm_with_trace(
                    prompts,
                    generation_name=generation_name
                )

                generated_code = self.extract_code(response_raw)
                
                # if the number of lines of the generated_code is less than 60, raise and error
                if not generated_code or len(generated_code.splitlines()) < 60:
                    error_message = (
                        "Generated code is not complete or not performing expected functionality. Please ensure to generate a complete sketch achieving the specified functionality."
                    )
                    self.logger.error(
                        self.get_session_id() + error_message
                    )
                    raise OSError(error_message)
          
                
                if not generated_code.strip() or not generated_code:
                    self.logger.error(
                        self.get_session_id()
                        + "No sketch code block found in LLM response."
                    )
                    error_message = "No sketch code block found in LLM response."
                    if attempt == max_retries - 1:
                        return False, generated_code, error_message
                    continue  # Go to next retry, hoping LLM formats correctly

                code = generated_code
                
                self.logger.info(
                    self.get_session_id()
                    + f"{'First attempt' if attempt == 1 else 'Error handling attempt NO.' + str(attempt)} code received. Executing..."
                )
                error = self.execute_code(
                    code, os.path.join(os.getcwd(), "compiling"), is_arduino=True
                )

                if not error:
                    self.logger.info(
                        self.get_session_id() + "Sketch compilation successful."
                    )
                    return True, code, ""
    
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


        return False, None, f"Max {max_retries} attempts exceeded. Last error from code execution: {error}"
 

    def extract_code(self, response):
        """
        Extracts the Arduino sketch code from the LLM response. Overrides the base class method.

        Args:
            response: The raw response from the LLM.

        Returns:
            Extracted code as a string.

        Raises:
            ValueError: If no supported code blocks are found.
        """
        return super().extract_code(response, language="cpp")

    def run(self):
        """
        Executes the sketch generation process by acquiring user input, filling specifications,
        and generating the sketch code.
        """
        print(
            "Model name: " + self.llm_strategy.model_name,
            "; Trace name: " + self.trace_name,
            "; Batch ID: " + self.session_id if self.session_id else "",
        )
        
        self.logger.info(
                self.get_session_id()
                + f"Starting {self.__class__.__name__} run with trace ID: {self.trace_id}"
            )
        
        try:
            
            self.get_user_input()
            self.logger.info(self.get_session_id() + "Sketch generation started.")
            self.dataset_summary()

            spec_signal, generated_sketch, error_spec = self.do_specification_filling(max_retries=5)
            
            if not spec_signal:
                self.logger.error(
                    self.get_session_id()
                    + f"Failed to fill specifications. Last error: {error_spec}"
                )
                self.raise_error_tracing(
                    RuntimeError(
                        f"Specification filling failed. Last error: {error_spec}"
                    )
                )
                self.trace.update(
                    output={
                        "status": "failed",
                        "last_code": generated_sketch,
                        "last_error": error_spec,
                    }
                )
                self.logger.error(
                    self.get_session_id()
                    + f"{self.__class__.__name__} run finished with failure."
                )
                
            else:
                self.app_specifications_filled = generated_sketch
                self.compose_spec_with_guideline_prompt()
                
                self.logger.info(
                    self.get_session_id() + f"Specification filling completed."
                )

                self.logger.info(
                    self.get_session_id()
                    + f"STARTED: First attempt to generate the sketch code."
                )
                status, generated_code, last_error = self.do_sketch_generation(max_retries=5)
                
                if status:
                    self.logger.info(
                        self.get_session_id()
                        + "Successfully generated and validated sketch code."
                    )
   
                    self.trace.update(
                        output={
                            "status": "Success",
                            "sketch": generated_code,
                        }
                    )
                    self.logger.info(
                        self.get_session_id()
                        + f"{self.__class__.__name__} run finished successfully."
                    )
                else:
                    self.logger.error(
                        self.get_session_id()
                        + f"Failed to generate valid code. Last error: {last_error}"
                    )
                    self.raise_error_tracing(
                    RuntimeError(f"Failed. Last error: {last_error}")
                )
                    self.trace.update(
                        output={
                            "status": "failed",
                            "last_code": generated_code,
                            "last_error": last_error,
                        }
                    )
                    self.logger.error(
                        self.get_session_id()
                        + f"{self.__class__.__name__} run finished with failure."
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
            raise  # Re-raise after handling
        except Exception as e:
            self.logger.error(
                self.get_session_id()
                + f"An unexpected error occurred outside of the loop: {traceback.format_exc()}"
            )
            self.trace.update(
                output={
                    "status": "failed",
                    "last_error": "An unexpected error occurred outside of the loop. " + str(e),
                }
            )
            self.raise_error_tracing(e)
            self.trace.update(output={"status": "failed", "last_error": str(e)})
            raise  # Re-raise after handling


# tmp test client
if __name__ == "__main__":
    import os

    from factories.llm_factory import LLMFactory

    # load_dotenv()
    # model_name = "gemini-1.5-flash"
    model_name = "phi4"

    llm_strategy = LLMFactory.create_llm("ollama", model_name=model_name)
    # llm_strategy = LLMFactory.create_llm("openai", model_name=model_name)

    # trace_id is used to stack the subtasks of a run together in LangFuse
    trace_id = str(uuid.uuid4()).split("-")[0]
    sketch_generator = SketchGenerator(
        llm_strategy,
        benchmark=False,
        trace_id=trace_id,
    )
    sketch_generator.run()
