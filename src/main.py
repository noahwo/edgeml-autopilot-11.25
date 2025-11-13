# %%
import ast
import os
import re
import subprocess
import sys
import time
import uuid

from dotenv import load_dotenv

# Add parent directory (Projects/) to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

 
project_path = os.path.dirname(__file__).split("/dev")[0]

# print(project_path)
sys.path.append(os.path.join(os.path.dirname(__file__), project_path))

sys.path.append(
    os.path.join(os.path.dirname(__file__), os.path.join(project_path, "src"))
)


from factories.llm_factory import LLMFactory
from processors.data_processor import DataProcessor
from processors.model_converter import ModelConverter
from processors.pysketch_generator import PySketchGenerator
from processors.sketch_generator import SketchGenerator
from processors.tpusketch_generator import TPUSketchGenerator

# %%


def clean_model_name(input_string):
    # Define the regex pattern to match ":<number>b" or ":latest" at the end of the string
    pattern = r"(:\d+b|:latest)$"
    pattern2 = r"(:latest)$"
    # Use re.sub to replace the matched pattern with an empty string
    parsed_string = re.sub(pattern2, "", input_string)

    return parsed_string


# %%
def run_batch_test(
    testee,
    trace_id,
    num_run,
    benchmarking,
    session_id,
    self_host_langfuse_flag,
    model_config
):
    load_dotenv()
    
    # openai_api_key = os.getenv("OPENAI_API_KEY")
    # model_name = "llama3.1:latest"
    # model_name = "llama3.1:70b"
    # model_name = "qwen2.5-coder:14b"
    # model_name = "phi4:latest"
    # model_name = "deepseek-r1:14b"
    # model_name = "gpt-4o"
    # model_name = "qwen2.5-coder:32b"
    
    # model_config = ("ollama", "qwen2.5-coder:32b", True)
    # model_config = ("openai", "gpt-4o-mini", False)

    
    llm_strategy = LLMFactory.create_llm(
        model_config[0], model_name=model_config[1], parameters=model_config[2]
    )
    # llm_strategy = LLMFactory.create_llm("openai", model_name=model_name)

    if session_id == "None":
        session_id = None
    else:
        session_id = clean_model_name(model_config[1]) + "_" + session_id
    print("Batch ID: ", session_id)

        # Pass trace_id to the processor
    processor_classes = {
        "data": DataProcessor,
        "convert": ModelConverter,
        "sketch": SketchGenerator,
        "pysketch": PySketchGenerator,
        "tpusketch": TPUSketchGenerator
    }

    processor_class = processor_classes.get(testee)
    if processor_class is not None:
        processor = processor_class(
            llm_strategy,
            trace_id=trace_id,
            num_run=num_run,
            benchmark=benchmarking,
            session_id=session_id,
            self_host_langfuse_flag=bool(self_host_langfuse_flag),
        )
    else:
        raise ValueError(f"Unknown testee type: {testee}")
    print(f"Running processor for {testee}:{session_id} with model config {model_config}")
    processor.run()
    

 
def main():
    
    num_runs = 30
    
    benchmarking = num_runs >= 20
    self_host_langfuse_flag = True
    # session_id = f"{stamp}_batch"

    stamp = str(uuid.uuid4()).split("-")[0][:4]
    # stamp = "da0a"

    # data: DataProcessor, convert: ModelConverter, sketch: SketchGenerator
    testee_list_ = {
        "convert": f"{stamp}_mc_batch",
        "data": f"{stamp}_dp_batch",
        "sketch": f"{stamp}_sg_batch",
        
        # "tpusketch": f"<stamp>_tpusg_batch",
        # "pysketch": f"<stamp>_psg_batch",
    }
    
    model_config_list = (
        # ("ollama", "qwen2.5-coder:32b", False),
        # ("ollama", "qwen2.5-coder:32b", True),
        
        # ("ollama", "phi4:latest", True),
        ("ollama", "phi4:latest", False),
        
        # ("ollama", "gemma3:27b", True),
        # ("ollama", "gemma3:27b", False),

        # ("ollama", "codestral:latest", True),
        # ("ollama", "codestral:latest", False),
        
        # ("ollama", "qwen2.5-coder:14b", True),
        # ("ollama", "qwen2.5-coder:14b", False),
        

        # ("ollama", "deepseek-coder-v2:16b", True),
        # ("ollama", "deepseek-coder-v2:16b", False),
 
        
    )
    time_start = time.time()
    for model_config in model_config_list:

    
        testee_list = testee_list_.copy()
        
        stamp_atm=str(uuid.uuid4()).split("-")[0][:4]
        for testee in testee_list:
            # Replace <stamp> with the actual stamp value
            testee_list[testee] = testee_list[testee].replace("<stamp>", stamp_atm)

        for testee, session_id in testee_list.items():
 
             
       

            for i in range(1, num_runs + 1):
                # for i in range(28, 31):

                trace_id = str(uuid.uuid4()).split("-")[0]

                print(
                    f"#{'='*40}#\n# Running batch test {i} of {num_runs} for {testee} #\n#{'='*40}#"
                )
                # Use subprocess to run the batch test in a separate process
                subprocess.run(
                    [
                        sys.executable,  # Use the current Python interpreter
                        __file__,  # The current script file
                        testee,  # Pass the testee type
                        trace_id,  # Pass the unique trace ID
                        str(i),  # Pass the current run index
                        str(benchmarking),  # Pass the benchmarking flag
                        session_id,
                        str(self_host_langfuse_flag),
                        str(model_config),  # Convert tuple to string
                    ]
                )

                if i < num_runs:
                    print(
                        f"Sleeping for 20 seconds to start the next batch: {testee}-{session_id}"
                    )
                    time.sleep(20)
 

if __name__ == "__main__":

    if len(sys.argv) > 1:
        # If arguments are passed, run the batch test with those arguments
        run_batch_test(
            testee=sys.argv[1],
            trace_id=sys.argv[2],
            num_run=int(sys.argv[3]),
            benchmarking=sys.argv[4] == "True",
            session_id=sys.argv[5],
            self_host_langfuse_flag=sys.argv[6] == "True",
            model_config=ast.literal_eval(sys.argv[7]),  # Safer than eval()

        )
    else:
        # Otherwise, run the main function
        main()



 
# %%
