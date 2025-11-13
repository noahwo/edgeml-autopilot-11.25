"""
LLMFactory Module

This module defines the LLMFactory class, responsible for creating instances
of different language model strategies based on the specified type.
"""

from base.llm_strategy import *


class LLMFactory:
    """
    LLMFactory is responsible for creating instances of language model strategies
    based on the provided type.
    """

    @staticmethod
    def create_llm(llm_type, **kwargs):
        """
        Creates an instance of an LLMStrategy based on the llm_type.

        Args:
            llm_type: Type of the language model ('langchain_openai', 'ollama', 'direct_openai').
            **kwargs: Additional keyword arguments required for the strategy.

        Returns:
            An instance of a subclass of LLMStrategy.

        Raises:
            ValueError: If the llm_type is unsupported.
        """
        parameters = kwargs.get("parameters", False)
        if llm_type == "openai":
            return OpenAILiteLLMStrategy(
                kwargs.get("model_name"),
                parameters,
            )
        # elif llm_type == "gemini":
        #     return GeminiLiteLLMStrategy(
        #         kwargs.get("model_name")
        #     )
        # elif llm_type == "openai_direct":
        #     return OpenAIStrategy(kwargs.get("api_key"), kwargs.get("model_name"))

        elif llm_type == "ollama":
            return OllamaLiteLLMStrategy(
                kwargs.get("model_name"),
                parameters,
            )

        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")
