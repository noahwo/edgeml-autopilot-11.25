"""!
BaseProcessor Module

This module defines the BaseProcessor abstract base class, which serves as the foundational class for all processors in the project. It provides shared functionalities such as environment setup, logging, LLM invocation with tracing, code execution, and error handling.
"""
#%%
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Union, final
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate

import colorlog
import pandas as pd
import requests
from dotenv import load_dotenv
from langfuse import Langfuse

try:
    # Try relative import first (for package usage)
    from .llm_strategy import (LLMStrategy, OllamaLiteLLMStrategy,
                               OpenAILiteLLMStrategy, OpenAIStrategy)
except ImportError:
    # Fall back to absolute import (for direct module execution)
    from llm_strategy import (LLMStrategy, OllamaLiteLLMStrategy,
                              OpenAILiteLLMStrategy)

# Third-party imports

 
 

class BaseProcessor(ABC):
    """!
    BaseProcessor is an abstract base class that provides common functionalities
    for all processor classes in the project.
    """

    def __init__(
        self,
        llm_strategy: LLMStrategy,
        trace_id,
        task_name,
        num_run=None,
        benchmark=False,
        session_id=None,
        self_host_langfuse_flag=False,
    ):
        """!
        Initializes the BaseProcessor with the given parameters.

        Args:
            llm_strategy: The language model strategy object providing invoke() method.
            trace_id: Unique identifier for tracing the whole run.
            num_run: Optional run number, to locate a run in a batch run.
            benchmark: Flag to indicate benchmarking(batch run).
            session_id: Optional batch identifier, to locate a batch test among different times of batch tests.
        """
        self.llm_strategy = llm_strategy
        self.load_environment()
        self.setup_logging()

        self.current_option = {}
        self.input_datatype = ""
        self.output_datatype = ""
        self.dataset_path = ""
        self.chat_history_key_value = ""
        self.dataset_summary_str = ""

        # For tracing
        self.trace_id = trace_id
        self.tags = [
            self.get_task_name(),
            "benchmark" if benchmark else "experiment",
            self.model_name,
        ]

        if session_id:
            if num_run is None:
                raise ValueError("num_run is required if session_id is provided")
            self.trace_metadata = {"num_run": num_run}
            self.session_id = session_id
        else:
            self.trace_metadata = {}
            self.session_id = None
        self.trace_name = self.trace_id[:4] + "_" + task_name

        self.self_host_langfuse_flag = self_host_langfuse_flag
        self.set_langfuse_trace()

    def get_session_id(self):
        """!
        Get the batch id for batch run.
        """
        if self.session_id:
            return f"{str(self.session_id)}|{str(self.trace_id)[:4]}_{self.get_task_name(short=True)}: "
        else:
            return f"{str(self.trace_id)}: "

    def set_langfuse_trace(
        self,
    ):

        trace_params = {
            "id": self.trace_id,
            "name": self.trace_name,
            "metadata": self.trace_metadata,
            "tags": self.tags,
        }
        if hasattr(self, "session_id"):
            trace_params["session_id"] = self.session_id

        if str(socket.gethostname()) == "! svm-125.cs.helsinki.fi":
            self.set_langfuse_trace_helper(
                "LANGFUSE_LOCAL_SECRET_KEY",
                "LANGFUSE_LOCAL_PUBLIC_KEY",
                "LANGFUSE_LOCAL_HOST",
            )
        elif self.self_host_langfuse_flag:
            self.set_langfuse_trace_helper(
                "LANGFUSE_LOCAL2_SECRET_KEY",
                "LANGFUSE_LOCAL2_PUBLIC_KEY",
                "LANGFUSE_LOCAL2_HOST",
            )
        else:
            self.set_langfuse_trace_helper(
                "LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_HOST"
            )
        langfuse = Langfuse()

        self.trace = langfuse.trace(**trace_params)

    def set_langfuse_trace_helper(self, arg0, arg1, arg2):
        """!
        Sets up the environment variables for Langfuse tracing.
        """
        
        secret_key = os.getenv(arg0)
        public_key = os.getenv(arg1)
        host = os.getenv(arg2)
        
        if secret_key is None or public_key is None or host is None:
            err_msg=f"Critical environment variables missing: {arg0}={secret_key}, {arg1}={public_key}, {arg2}={host}"
            self.logger.error(err_msg)
            raise ValueError(err_msg)

        os.environ["LANGFUSE_SECRET_KEY"] = secret_key
        os.environ["LANGFUSE_PUBLIC_KEY"] = public_key
        os.environ["LANGFUSE_HOST"] = host

    def get_task_name(self, short=False):
        """
        Returns the task name based on the class name.

        Args:
            short: If True, returns abbreviated task name (first letters of each word).
                If False, returns full task name in snake_case.

        Returns:
            Task name as a string.
        """
        # Convert class name from CamelCase to snake_case
        class_name = self.__class__.__name__
        task_name = "".join(
            ["_" + c.lower() if c.isupper() else c for c in class_name]
        ).lstrip("_")

        if not short:
            return task_name

        # Get first letter of each part for short version
        return "".join(part[0].lower() for part in task_name.split("_"))

    def load_environment(self):
        """!
        Loads environment variables from the .env file and sets up necessary environment settings.
        """
        load_dotenv()
        if (
            self.llm_strategy is OpenAILiteLLMStrategy
            or self.llm_strategy is OpenAIStrategy
        ):
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.model_name = self.llm_strategy.model_name

        # self.set_langchain_env()

    def set_langchain_env(self):
        os.environ["LANGCHAIN_API_KEY"] = str(os.getenv("LANGCHAIN_API_KEY"))
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_PROJECT"] = "default"

    @final
    def setup_logging(self):
        """!
        Sets up logging with both console and file handlers using colorlog for colored console output.
        """
        self.logger = colorlog.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        console_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s %(levelname)s %(message)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )

        file_formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        # make sure the log file is created
        if not os.path.exists(f"logs/{self.__class__.__name__}.log"):
            os.makedirs("logs", exist_ok=True)
        file_handler = logging.FileHandler(f"logs/{self.__class__.__name__}.log")
        console_handler = colorlog.StreamHandler()

        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)

        file_handler.setLevel(logging.INFO)
        console_handler.setLevel(logging.INFO)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.propagate = False

    def get_network_latency(self, url: str, latency: float = 0) -> float|int:
        """!
        Measures the network latency to the specified URL.

        Args:
            url: The URL to measure latency to.

        Returns:
            Latency in seconds, or -1.0 if measurement fails.
        """
        if url == -2:
            return int(url)
        try:
            # [ ] NOTE: this is not accurate, it mostly measure the local execution time
            start_time = time.time()
            response = requests.get(url)
            end_time = time.time()
            latency = end_time - start_time
            self.logger.info(
                self.get_session_id() + f"Network latency to {url}: {latency} seconds"
            )
            return latency
        except requests.RequestException as e:
            self.logger.error(
                self.get_session_id() + f"Network latency measurement failed: {e}"
            )
            return -1.0  # Indicates failure

    @final
    def invoke_llm_with_trace(
        self, prompts: list[str|PromptTemplate] | list[str], generation_name: str, lib_extract=False
    ):
        """!
        Invokes the LLM with tracing, tagging network latency to the trace.
        Network latency is measured before the trace and added as metadata.
        A trace is a whole run, can include multiple generations/subtasks. Generation is the term for langfuse, subtask is the term for the framework. A generation is a subtask, a component in a stage.

        Args:
            prompt: The prompt to send to the LLM.
            trace_name: Task name for the series of tracing, with identifiers.
            generation_name: Subtask/generation name for tracing, task_name.
        Returns:
            A dictionary containing the LLM's response.
        """

        # Measure network latency before invoking the LLM
        llm_endpoint = self.llm_strategy.get_endpoint_url()
        network_latency = self.get_network_latency(llm_endpoint)

        if network_latency < 0:
            network_latency = None  # Handle latency measurement failure
        else:
            self.trace_metadata["network_latency"] = network_latency

        metadata_ = {
            "existing_trace_id": self.trace_id,
            # "session_id": getattr(self, "session_id", None),
            # "trace_id": self.trace_id,
            # "trace_name": self.trace_name,
            "generation_name": generation_name,
            "trace_metadata": self.trace_metadata,
            # "tags": self.tags,
            # "debug_langfuse": True, THIS CAUSES INFINITE RECURSION ERROR
        }
        # metadata_ = {
        #     "session_id": getattr(self, "session_id", None),
        #     "trace_id": self.trace_id,
        #     "trace_name": self.trace_name,
        #     "generation_name": generation_name,
        #     "trace_metadata": self.trace_metadata,
        #     "tags": self.tags,
        #     # "debug_langfuse": True, THIS CAUSES INFINITE RECURSION ERROR
        # }
        strategy_to_model = {
            OpenAILiteLLMStrategy: "gpt-4o-mini",
            OllamaLiteLLMStrategy: "granite3.1-dense:2b",
        }

        lib_extract_model = (
            strategy_to_model.get(type(self.llm_strategy)) if lib_extract else None
        )

        return self.llm_strategy.invoke(prompts, metadata_, lib_extract_model=lib_extract_model)

        # return self.llm_strategy.invoke(prompts, metadata_)
        
        
    def raise_error_tracing(self, e):
        """!
        Raises an error under a trace within the run to mark the failure in LangSmith.
        0 cost to token and time.

        Args:
            e: The exception to raise.
        """
        if not hasattr(e, "_logged_to_tracing"):
            task_name = self.get_task_name()
            short_task_name = self.get_task_name(short=True)
            short_id = self.trace_id[:2]
            self.trace.span(
                level="ERROR",
                name="error_"
                + short_id
                + "_"
                + short_task_name
                + "_failure_signal_"
                + task_name,
                status_message=str(e),
            )
            e._logged_to_tracing = True

    # @final
    # def update_chat_history(self, prompt, response):
    #     """!
    #     Updates the chat history with the latest prompt and response.

    #     Args:
    #         prompt: The prompt sent to the LLM.
    #         response: The raw response received from the LLM.
    #     """
    #     self.chat_history = f'"most_recent_round_of_conversation":[HumanMessage({str(prompt)}),AIMessage({str(response)})]'

    @abstractmethod
    def get_user_input(self):
        """!
        Abstract method to acquire user input. Must be implemented by subclasses.
        """
        pass

    @final
    def dataset_summary(self) -> str:
        """!
        Generates a summary of the dataset including shape, statistics, and column names.
        To give the inspiration to the LLM.

        Returns:
            A string summarizing the dataset.

        Raises:
            Exception: If reading the dataset fails.
        """
        try:
            dataframe = pd.read_csv(self.dataset_path)
            description = dataframe.describe().to_dict()
            head = dataframe.head().to_dict()

            # Create a simple string representation without dictionary formatting
            summary_str = (
                f"Dataset shape: {dataframe.shape}\n"
                f"Dataset descriptive statistics: {description}\n"
                f"Dataset first 5 rows: {head}\n"
                f"Column names: {list(dataframe.columns)}\n"
                f"Note: column names are case sensitive, remember that."
            )
            summary_str = summary_str.replace("{", "{{").replace("}", "}}")
            self.dataset_summary_str = summary_str

            self.logger.info(
                self.get_session_id()
                + f"Dataset summary generated for {self.dataset_path}"
            )
            return self.dataset_summary_str

        except Exception as e:
            self.logger.error(
                self.get_session_id()
                + f"Error generating dataset summary for {self.dataset_path}: {e}"
            )
            raise e

    @abstractmethod
    def run(self):
        """!
        Abstract method to execute the processor. Must be implemented by subclasses.
        """
        pass

    def prepare_execution_environment(
        self, validation_workspace, is_arduino=False, fixed_timestamp=None
    ):
        """
        Prepare paths and directories for code execution/compilation.
        Creates 1) working paths, 2) validated paths, and 3) execution command.

        Args:
            validation_workspace: Base directory for execution
            is_arduino: Whether to prepare for Arduino (.ino) or Python execution
            fixed_timestamp: ONLY for testing, if provided, use this timestamp instead of the current time

        Returns:
            tuple: (work_dir, tmp_file, valid_dir, valid_file, execute_command)
        """
        timestamp = (
            datetime.now().strftime("%Y%m%d%H%M%S")
            if fixed_timestamp is None
            else fixed_timestamp
        )

        # Get file paths based on language type and timestamp
        work_dir, tmp_file = self._create_working_paths(
            validation_workspace, timestamp, is_arduino
        )
        valid_dir, valid_file = self._create_validated_paths(
            validation_workspace, timestamp, is_arduino
        )

        # Set up the execution command and any special requirements
        execute_command = self._create_execution_command(
            work_dir, tmp_file, validation_workspace, is_arduino
        )

        return work_dir, tmp_file, valid_dir, valid_file, execute_command

    # [x] BUG: Servere bugs in the following paths creation functions
    def _create_working_paths(self, validation_workspace, fixed_timestamp, is_arduino):
        """
        Create working directory and temporary file paths.

        Args:
            validation_workspace: Base directory
            fixed_timestamp: Current timestamp string
            is_arduino: Whether this is for Arduino code

        Returns:
            tuple: (work_dir, tmp_file). E.g., for task_name="test-task", model_name="gpt-3.5-turbo", timestamp="20210101120000", the paths for python code will be:
            work_dir = "tmp_20210101120000_test-task_gpt-3.5-turbo"
            tmp_file = "tmp_20210101120000_test-task_gpt-3.5-turbo.py". for arduino code, the paths will be:
            work_dir = "compiling_20210101120000_gpt-3.5-turbo"
            tmp_file = "compiling_20210101120000_gpt-3.5-turbo.ino"

        """

        file_ext = ".ino" if is_arduino else ".py"

        work_dir_name = (
            f"compiling_{fixed_timestamp}_{self.llm_strategy.model_name}"
            if is_arduino
            else f"tmp_{fixed_timestamp}_{self.get_task_name(short=True)}_{self.llm_strategy.model_name}"
        )

        # Create working directory
        work_dir = os.path.join(validation_workspace, work_dir_name)
        os.makedirs(work_dir, exist_ok=True)

        # Create temporary file path
        # self.logger.info(f"TEST PRINT work_dir: {work_dir}")
        tmp_file = os.path.join(work_dir, f"{work_dir_name}{file_ext}")
        # self.logger.info(
        #     f"TEST PRINT\n tmp_file is work_dir \n{work_dir}\n + tmp_file name \n{str(work_dir_name)}{file_ext}\nwhich is \n{tmp_file}"
        # )
        return work_dir, tmp_file

    def _create_validated_paths(
        self, validation_workspace, fixed_timestamp, is_arduino
    ):
        """
        Create validated directory and file paths for storing validated code.

        Args:
            validation_workspace: Base directory
            fixed_timestamp: Current timestamp string
            is_arduino: Whether this is for Arduino code

        Returns:
            tuple: (valid_dir, valid_file)
        """
        file_ext = ".ino" if is_arduino else ".py"

        valid_dir_name = (
            f"valid_{fixed_timestamp}_{self.llm_strategy.model_name}"
            if is_arduino
            else f"valid_{fixed_timestamp}_{self.get_task_name(short=True)}_{self.llm_strategy.model_name}"
        )

        valid_dir = os.path.join(validation_workspace, valid_dir_name)
        valid_file = os.path.join(valid_dir, f"{valid_dir_name}{file_ext}")

        return valid_dir, valid_file

    def _create_execution_command(
        self, work_dir, tmp_file, validation_workspace, is_arduino
    ):
        """
        Prepare the execution command and handle any special setup requirements.

        Args:
            work_dir: Working directory path
            tmp_file: Temporary file path
            validation_workspace: Base directory
            is_arduino: Whether this is for Arduino code

        Returns:
            list: Command to execute as list of strings
        """
        if is_arduino:
            # Copy required model.h file for Arduino
            shutil.copy(
                os.path.join(validation_workspace, "model.h"),
                os.path.join(work_dir, "model.h"),
            )
            execute_command = [
                "arduino-cli",
                "compile",
                "--fqbn",
                "arduino:mbed:nano33ble",
                tmp_file,
            ]
        else:
            execute_command = [sys.executable, tmp_file]

        return execute_command

    def execute_code(
        self, code, validation_workspace, is_arduino=False, local_retry=False, remote_execution=False
    ):
        """!
        Execute the code snippet and return the error message if any.

        Args:
            code (str): The code snippet to execute.
            validation_workspace (str): The path to the workspace directory.
            is_arduino (bool): Indicates if the code is an Arduino sketch (.ino) file.
            local_retry (bool): Whether this is a retry after installing dependencies.
            remote_execution (bool): Whether to execute remotely via SSH file transfer.

        Returns:
            str: The error message if any.
            None: If the code executed successfully.

        """
        # Handle remote execution via SSH file transfer
        if remote_execution:
            return self._execute_code_remotely(code, validation_workspace)
        
        # Continue with local execution
        work_dir, tmp_file, valid_dir, valid_file, execute_command = (
            self.prepare_execution_environment(validation_workspace, is_arduino)
        )

        os.makedirs(os.path.dirname(tmp_file), exist_ok=True)

        try:
            with open(tmp_file, "w") as file:
                file.write(code)
        except (IOError, OSError) as e:
            self.logger.error(
                self.get_session_id() + f"Failed to write code to file {tmp_file}: {e}"
            )
            self._cleanup_execution_files(work_dir, tmp_file)
            return str(e)

        self.logger.info(
            self.get_session_id() + "Executing/compiling the code snippet..."
        )

        try:
            result = subprocess.run(execute_command, capture_output=True, text=True)
            has_error = self.has_error_in_output(result)

            if result.returncode == 0 and not has_error:
                self._handle_successful_execution(result, code, valid_file)
                self._cleanup_execution_files(work_dir, tmp_file)
                return None
            else:
                # If there's an error, handle it
                error_message = result.stderr or result.stdout
                return self._handle_execution_error(
                    error_message,
                    code,
                    validation_workspace,
                    work_dir,
                    tmp_file,
                    local_retry,
                    is_arduino,
                )
        except Exception as e:
            self.logger.error(
                self.get_session_id() + f"Exception during code execution: {e}"
            )
            traceback.print_exc()
            self._cleanup_execution_files(work_dir, tmp_file)
            return str(e)

    def _execute_code_remotely(self, code, validation_workspace):
        """
        Execute code remotely via direct SSH file transfer and execution.
        
        This method:
        1. Creates a temporary Python file locally
        2. Transfers it to the remote machine via SCP
        3. Executes it remotely via SSH with real-time output streaming
        4. Captures stdout, stderr, and traceback information
        5. Cleans up the temporary file on the remote machine
        
        Args:
            code (str): The Python code to execute remotely
            validation_workspace (str): Path to the workspace directory (unused for direct transfer)
            
        Returns:
            str: Error message if execution failed, None if successful
        """
        # Configuration for remote machine
        remote_host = os.getenv('REMOTE_HOST', 'coral')
        remote_path = os.getenv('REMOTE_EXEC_PATH', '/home/mendel/tinyml_autopilot/tmp')
        remote_env_activate = os.getenv('REMOTE_PYTHON_ENV', 'source ~/tinyml_autopilot/tinyml-env/bin/activate')
        remote_python = os.getenv('REMOTE_PYTHON_EXECUTABLE', '/home/mendel/tinyml_autopilot/tinyml-env/bin/python')      

        # Generate unique script name with timestamp
        script_id = f"script_{uuid.uuid4().hex[:8]}_{int(time.time())}"
        script_filename = f"{script_id}.py"
        remote_script_path = f"{remote_path}/{script_filename}"
        
        try:
            check_result = subprocess.run(['ssh', remote_host, "cat /sys/class/apex/apex_0/device_owner"], capture_output=True, text=True, timeout=10)
            if check_result.returncode == 0:
                owner_pid = check_result.stdout.strip()
                self.logger.info(self.get_session_id() + f"TPU Device owner PID: {owner_pid}")
                # If the owner PID is not empty and not "0", kill the process
                if owner_pid and owner_pid != "0":
                    self.logger.warning(self.get_session_id() + f"TPU Device owned by PID {owner_pid}, killing process")
                    kill_cmd = ['ssh', remote_host, f'kill {owner_pid} || true']
                    subprocess.run(kill_cmd, capture_output=True, text=True, timeout=10)
        except:
            pass 
        
        try:
            # Create temporary local file with the code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                temp_file.write(code)
                local_script_path = temp_file.name
            
            # Save a copy for future analysis
            analysis_dir = os.path.join("tmp")
            os.makedirs(analysis_dir, exist_ok=True)
            analysis_file_path = os.path.join(analysis_dir, f"{script_id}.py")
            shutil.copy2(local_script_path, analysis_file_path)
            
            # self.logger.info(self.get_session_id() + f"Created local temp script: {local_script_path}")
            
            # Step 0: Commands making
            
            # To create remote directory if it doesn't exist 
            mkdir_command = [
                'ssh',  remote_host,
                f'mkdir -p {remote_path}'
            ]
            
            # To transfer the script to the remote machine
            scp_command = [
                'scp', local_script_path,
                f'{remote_host}:{remote_script_path}'
            ]
            
            # To execute the script remotely
            ssh_command = [
                'ssh',  remote_host ,
                f'cd {remote_path} && {remote_python} {script_filename}'
            ]
            
            # To cleanup the remote script after execution
            cleanup_command = [
                'ssh',remote_host, f'rm -f {remote_script_path}'
            ]
            
            # Kill leftover processes if any
            kill_command = [
                'ssh', remote_host,
                f'pkill -f {script_id} || true'  # Ignore if no process found
            ]
            
            # Step 1: Create remote directory if it doesn't exist
            
            self.logger.info(self.get_session_id() + f"Creating remote directory: {remote_path}")
            mkdir_result = subprocess.run(mkdir_command, capture_output=True, text=True, timeout=30)
            
            if mkdir_result.returncode != 0:
                error_msg = f"Failed to create remote directory: {mkdir_result.stderr}"
                self.logger.error(self.get_session_id() + error_msg)
                return error_msg
            
            # Step 2: Transfer script to remote machine
            
            self.logger.info(self.get_session_id() + f"Transferring script to remote: {remote_script_path}")
            scp_result = subprocess.run(scp_command, capture_output=True, text=True, timeout=60)
            
            if scp_result.returncode != 0:
                error_msg = f"Failed to transfer script: {scp_result.stderr}"
                self.logger.error(self.get_session_id() + error_msg)
                return error_msg
            
            # Step 3: Execute script remotely with real-time output streaming
            
            self.logger.info(self.get_session_id() + f"Executing script remotely: {script_filename}")
            
            # Execute with real-time output capture
            execution_error = self._stream_remote_execution(ssh_command, script_id)
            
            # Step 4: Cleanup remote file
            
            self.logger.info(self.get_session_id() + f"Cleaning up remote script: {remote_script_path}")
            # kill leftover processes if any
            kill_process_result = subprocess.run(kill_command, capture_output=True, text=True, timeout=30)
            if kill_process_result.returncode != 0:
                self.logger.warning(self.get_session_id() + f"Failed to kill leftover processes: {str(kill_process_result)}")
                
            cleanup_result = subprocess.run(cleanup_command, capture_output=True, text=True, timeout=30)
            
            if cleanup_result.returncode != 0:
                self.logger.warning(self.get_session_id() + f"Failed to cleanup remote file: {cleanup_result.stderr}")
            
            return execution_error
            
        except subprocess.TimeoutExpired as e:
            error_msg = f"Remote execution timeout: {e}"
            self.logger.error(self.get_session_id() + error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Remote execution failed: {e}"
            self.logger.error(self.get_session_id() + error_msg)
            traceback.print_exc()
            return error_msg
        finally:
            # Cleanup local temporary file
            try:
                if 'local_script_path' in locals():
                    os.unlink(local_script_path)
                    self.logger.debug(self.get_session_id() + f"Cleaned up local temp file: {local_script_path}")
            except Exception as cleanup_error:
                self.logger.warning(self.get_session_id() + f"Failed to cleanup local temp file: {cleanup_error}")
    
    def _stream_remote_execution(self, ssh_command, script_id):
        """
        Execute SSH command with real-time output streaming to local CLI.
        
        Args:
            ssh_command (list): SSH command to execute
            script_id (str): Unique script identifier for logging
            
        Returns:
            str: Error message if execution failed, None if successful
        """
        try:
            # Start the SSH process with streaming output
            process = subprocess.Popen(
                ssh_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0,  # Unbuffered for immediate output
                universal_newlines=True
            )
            
            self.logger.info(self.get_session_id() + f"🚀 Remote execution started for {script_id}")
            
            # Real-time output streaming with timeout
            stdout_lines = []
            stderr_lines = []
            
            # Use threading for reliable output capture
            import threading
            import queue
            
            def read_stdout():
                try:
                    if process.stdout:
                        for line in iter(process.stdout.readline, ''):
                            if line:
                                line = line.rstrip('\n')
                                stdout_lines.append(line)
                                # print(f"📡 [REMOTE] {line}")
                    if stdout_lines:
                        if 'Error' in stdout_lines:
                            stderr_lines= stdout_lines
                        else:
                            self.logger.info(self.get_session_id() + f"[STDOUT] {stdout_lines}")
                except:
                    pass
            def read_stderr():
                try:
                    if process.stderr:
                        for line in iter(process.stderr.readline, ''):
                            if line:
                                line = line.rstrip('\n')
                                stderr_lines.append(line)
                                # print(f"📡 [REMOTE ERROR] {line}")
                                # self.logger.error(self.get_session_id() + f"[STDERR] {line}")
                    if stderr_lines and 'Traceback (most recent call last)' not in stderr_lines:
                        self.logger.error(self.get_session_id() + f"[STDERR] {stderr_lines}")
                except:
                    pass
                    pass
            
            # Start reader threads
            stdout_thread = threading.Thread(target=read_stdout)
            stderr_thread = threading.Thread(target=read_stderr)
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            
            stdout_thread.start()
            stderr_thread.start()
            
            # Wait for process completion with timeout
            timeout_seconds = 60  # 1 minute timeout
            start_time = time.time()
            
            while True:
                # Check if process has finished
                if process.poll() is not None:
                    break
                
                # Check for overall timeout
                current_time = time.time()
                if current_time - start_time > timeout_seconds:
                    self.logger.error(self.get_session_id() + f"Remote execution timeout after {timeout_seconds} seconds")
                    process.terminate()
                    time.sleep(2)  # Give it time to terminate gracefully
                    if process.poll() is None:
                        process.kill()  # Force kill if still running
                    return f"Remote execution timeout after {timeout_seconds} seconds"
                
                # Sleep briefly to prevent excessive CPU usage
                time.sleep(0.1)
            
            # Wait for threads to finish reading all output
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)
            
            # Get the final exit code
            exit_code = process.returncode
            full_stdout = '\n'.join(stdout_lines)
            full_stderr = '\n'.join(stderr_lines)
            
            # Always return the combined output, regardless of success/failure
            combined_output = ""
            if full_stdout:
                combined_output += full_stdout
            if full_stderr:
                if combined_output:
                    combined_output += "\n"
                combined_output += full_stderr
            
            if exit_code == 0:
                self.logger.info(self.get_session_id() + f"✅ Remote execution completed successfully for {script_id}")
                # print(f"✅ Remote execution completed successfully!")
                # Return None for success (no error), but actual output is already logged/printed
                return None
            else:
                # For errors, return the combined output so the calling code can process it
                error_msg = f"❌ Remote execution failed with exit code {exit_code}"
                if full_stderr:
                    # Check for Python traceback in stderr
                    if 'Traceback' in full_stderr:
                        self.logger.error(self.get_session_id() + "Python traceback detected in remote execution:")
                        self.logger.error(self.get_session_id() + "🔍 [REMOTE TRACEBACK] \n"+full_stderr)
                        # print("🔍 [REMOTE TRACEBACK]")
                        # for line in full_stderr.split('\n'):
                        #     if line.strip():
                        #         print(f"🔍   {line}")
                
                self.logger.error(self.get_session_id() + error_msg)
                # print(f"❌ Remote execution failed!")
                # Return the actual error output, not just a generic message
                return combined_output if combined_output else error_msg
                
        except Exception as e:
            error_msg = f"Failed to stream remote execution: {e}"
            self.logger.error(self.get_session_id() + error_msg)
            print(f"💥 Remote execution streaming failed: {e}")
            return error_msg
    
    def _handle_execution_error(
        self,
        error_message,
        code,
        validation_workspace,
        work_dir,
        tmp_file,
        local_retry,
        is_arduino,
    ):
        """
        Handle errors that occur during code execution.

        Args:
            error_message: The error output from execution
            code: The executed code
            validation_workspace: Path to the workspace directory
            work_dir: Working directory for execution
            tmp_file: Path to the temporary file
            local_retry: Whether this is already a retry
            is_arduino: Whether this is Arduino code

        Returns:
            str: Error message, or None if error was resolved
        """
        if not local_retry and is_arduino:
            # This try-except is for avoiding the error in bigger workflow when the library is not installed
            self._cleanup_execution_files(work_dir, tmp_file)
            try:
                if missing_lib_str := self._whether_missing_lib_in_error_message(
                    error_message
                ):
                    return self._try_install_missing_library(
                        missing_lib_str, code, validation_workspace, error_message
                    )
            except Exception as e_lib_install:
                self.logger.info(
                    self.get_session_id()
                    + f"Arduino library installation failed, exception: {e_lib_install}. Ignore and continue the program."
                )

        self._cleanup_execution_files(work_dir, tmp_file)
        return error_message

    def _handle_successful_execution(self, result, code, valid_file):
        """
        Handle successful code execution by logging output and saving the code.

        Args:
            result: The subprocess.run result
            code: The executed code
            valid_file: Path where the validated code should be saved
        """
        if result.stdout and result.stdout.strip():
            self.logger.info(
                self.get_session_id() + f"Program output: {result.stdout.strip()}"
            )

        # Create new directory based on save_file name (without suffix)
        os.makedirs(os.path.dirname(valid_file), exist_ok=True)
        with open(valid_file, "w") as file:
            file.write(code)

    def _cleanup_execution_files(self, work_dir, tmp_file):
        """
        Clean up temporary execution files and directories.

        Args:
            work_dir: Directory to remove
            tmp_file: File to remove
        """
        try:
            if os.path.exists(work_dir):
                self.logger.debug(f"Attempting to remove directory: {work_dir}")
                shutil.rmtree(work_dir, ignore_errors=True)
                self.logger.info(f"Successfully removed directory: {work_dir}")
            if os.path.exists(tmp_file):
                self.logger.debug(f"Attempting to remove file: {tmp_file}")
                os.remove(tmp_file)
                self.logger.info(f"Successfully removed file: {tmp_file}")
        except Exception as e:
            self.logger.warning(f"Cleanup failed: {e}")

    def _try_install_missing_library(
        self, missing_lib_str, code, validation_workspace, og_error_message
    ):
        """
        Attempts to install a missing library and retry code execution.

        Args:
            missing_lib_str: The name of the missing library detected in error message
            code: The code to retry executing after library installation
            validation_workspace: The workspace directory

        Returns:
            Error message if installation or retry fails, None if successful
        """
        self.logger.info(
            self.get_session_id()
            + f"{missing_lib_str} related library is missing, searching and installing..."
        )

        library_name = self.search_install_arduino_library(missing_lib_str)
        if library_name is None:
            return f"{og_error_message}\nLibrary name not found for {missing_lib_str}"

        self.logger.info(
            self.get_session_id()
            + f'Installing library, name: "{library_name}" (given by SLM)'
        )

        # Install the library
        result = subprocess.run(
            ["arduino-cli", "lib", "install", library_name.replace('"', "")],
            capture_output=True,
            text=True,
        )
        combined_output = (result.stdout or "") + (result.stderr or "")
        has_error = self.has_error_in_output(
            result,
            indicators=["Error installing"],
        )

        if has_error:
            self.logger.error(
                self.get_session_id()
                + f"Arduino library installation failed: {combined_output}"
            )
            return f"{og_error_message}\n{combined_output}"
        else:
            self.logger.info(
                self.get_session_id()
                + f"Arduino library installation succeeded, result(raw): {result}, retrying the program..."
            )
            return self.execute_code(
                code,
                validation_workspace,
                is_arduino=True,
                local_retry=True,
            )

    def _whether_missing_lib_in_error_message(self, error_message: str) -> str | bool:
        pattern = r"fatal error: (.*?): No such file or directory"
        match = re.search(pattern, error_message)
        return match.group(1) if match else False

    def search_install_arduino_library(self, missing_lib_str: str) -> str | None:
        """!
        Search and install the Arduino library based on the error message.
        """
        # 2. Remove .h if it exists at the end
        missing_lib = re.sub(r"\.h$", "", missing_lib_str)
        # 3. Replace underscores with spaces
        lib_name = missing_lib.replace("_", " ")
        # 4. search the library in the arduino library manager
        execute_command = [
            "arduino-cli",
            "lib",
            "search",
            "--names",
            lib_name,
        ]
        unfiltered_list_lib_search_raw = subprocess.run(
            execute_command, capture_output=True, text=True
        )
        unfiltered_list_lib_search = unfiltered_list_lib_search_raw.stdout
        print("#########################")
        print("STDOUT OF ARDUINO LIB SEARCH:")
        print(unfiltered_list_lib_search)
        print("#########################")
        if (
            "No libraries matching your search" in unfiltered_list_lib_search
            or unfiltered_list_lib_search_raw.returncode == 1
        ):
            return None
        list_lib_search = "\n".join(
            line
            for line in unfiltered_list_lib_search.splitlines()
            if line.startswith("Name:")
        )

        self.logger.info(
            self.get_session_id()
            + f"Arduino library search results:\n{list_lib_search}"
        )

        lib_prompt = f"""Based on the search results of a missing arduino library named "{lib_name}", pick the most suitable one from the list, and output the clean string value of the name. Your output should exactly match one and only one from the following double quoted names. Return "None" if search results are empty:\n### SEARCH RESULTS\n```{list_lib_search}```\n### YOUR ANSWER OF A CLEAN STRING\n```\n```
        """
        response = self.invoke_llm_with_trace(
            lib_prompt,
            generation_name=f"{self.trace_id[:2]}_{self.get_task_name(short=True)}_search_arduino_lib",
            lib_extract=True,
        )
        return str(response).strip()

    def extract_code(
        self, response: str | object, language: str = "python"
    ) -> str | None:
        """!
        Extract code from the LLM response, handling different output formats.
        If no Python code blocks, try to extract from any code blocks
        If no code blocks, raise exception as not supported output format

        Args:
            response: The raw response from the LLM.
            language: Target language to extract ("python" or "cpp")

        Returns:
            Extracted code as a string.

        Raises:
            ValueError: If no supported code blocks are found.
        """
        content = str(response)

        if language == "cpp":
            return self.extract_code_cpp(content)
        elif language == "python":
            # First try explicit Python blocks
            pattern = r"```python\s*(.*?)\s*```"
            if match := re.search(pattern, content, re.DOTALL):
                code = match.group(1).strip()
                if code.startswith("{") and code.endswith("}"):
                    # remove the starting parts of { "somthing: " and the tail "}
                    code = code[code.index(":") + 1 : -1].strip()
       
                return str(code)
                    

            # Then try generic code blocks
            pattern = r"```\s*(.*?)\s*```"
            if match := re.search(pattern, content, re.DOTALL):
                code= match.group(1).strip()
                if code.startswith("{") and code.endswith("}"):
                    # remove the starting parts of { "somthing: " and the tail "}
                    code = code[code.index(":") + 1 : -1].strip()
       
                return str(code)

            raise ValueError("The output is not a valid code block. Follow the instructions in the prompt return a complete, correct Python script enclosed in a single ```python\n<generated_code>\n``` block.")

    # attributes is a dictionary like {"key1": "value1", "key2": "value2"}, i need to search in code line by line if there is any line containing "path_to" or "path to", is so, try to match if there is any keys in the dictionary attributes that are being the starting string in that line, then replace that line to be like key1 = "value1". Loop this for every line in code parameter. The lines cloud be seperated by newlines or "\n".
    # replacement for the same key can be check only once
    
    def check_placeholders(self, code, attributes):
        """
        Check and replace placeholders in the code with actual values from attributes.

        Args:
            code (str): The code snippet to check.
            attributes (dict): Dictionary of attributes to replace in the code.

        Returns:
            str: Code with placeholders replaced by actual values.
        """
        lines = code.splitlines()
        replaced_keys = set()  # Track keys that have been replaced
        
        for i, line in enumerate(lines):
            
            for key, value in attributes.items():
                # Skip if this key has already been replaced
                if key in replaced_keys:
                    continue
                    
                if "{key} =" in line:
                    lines[i] = f"{key} = \"{value}\""
                    print(
                        self.get_session_id()
                        + f"Replaced placeholder '{key}' with value '{value}'"
                    )
                    replaced_keys.add(key)  # Mark key as replaced
                    break  # Move to next line after replacement
                    
                elif (key.split("_")[0] == line.split("_")[0]) or (key.split("_")[0].upper() == line.split("_")[0]):
                    lines[i] = f'{line.split("=")[0]} = \"{value}\"' if isinstance(value, str) else f'{line.split("=")[0]} = {value}'
                    replaced_keys.add(key)  # Mark key as replaced
                    break  # Move to next line after replacement
                
        return "\n".join(lines)
        

        
    def extract_code_cpp(self, content):
        matches = list(re.finditer(r"```(?:cpp|ino)\n(.*?)```", content, re.DOTALL))

        # Combine all found code blocks
        code_blocks = [match.group(1).strip() for match in matches]
        if not code_blocks:
            raise ValueError("Unsupported output format: no cpp/ino code blocks found")
        match_count = len(matches)
        # print(f"Found {match_count} cpp/ino code blocks.")
        # Join all code blocks with newlines
        combined_code = "\n\n".join(code_blocks)

        self.logger.info(
            self.get_session_id()
            + f"Found {match_count} cpp/ino code blocks, concatenated if possible."
        )
        return combined_code

    def has_error_in_output(self, result, indicators=None):
        """
        Check if output contains error indicators.

        Args:
            result: subprocess.run result
            indicators: List of error indicator strings to look for

        Returns:
            bool: True if any error indicators found
        """
        combined_output = (result.stdout or "") + (result.stderr or "")
        if indicators is None:
            indicators = ["error:", "exception:", "traceback:", "failed:"]

        return any(indicator in combined_output.lower() for indicator in indicators)
