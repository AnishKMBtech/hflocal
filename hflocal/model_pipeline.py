import os
from transformers import pipeline, AutoTokenizer


class ModelPipeline:
    """
    A wrapper around Hugging Face's pipeline that adds save/load support.

    This is the primary way to download a pipeline model from Hugging Face
    and save it locally for offline use.
    """

    def __init__(self, model_name_or_path, task):
        """
        Initialize the model pipeline.

        Args:
            model_name_or_path (str): The HF model name (e.g. 'gpt2') or a local path.
            task (str): The pipeline task (e.g. 'text-generation', 'fill-mask',
                        'sentiment-analysis', 'question-answering').
        """
        try:
            print(f"Loading pipeline (task='{task}', model='{model_name_or_path}')...")
            self.pipe = pipeline(task, model=model_name_or_path)
            self.task = task
            print("Pipeline ready.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to create pipeline for task='{task}' with model='{model_name_or_path}'.\n"
                f"Check that the model name is correct and compatible with the task.\n"
                f"Original error: {e}"
            ) from e

    def __call__(self, *args, **kwargs):
        """
        Run inference through the pipeline.

        Args:
            *args: Positional arguments forwarded to the pipeline.
            **kwargs: Keyword arguments forwarded to the pipeline.

        Returns:
            The pipeline inference result.
        """
        return self.pipe(*args, **kwargs)

    def save(self, save_directory):
        """
        Save the pipeline's model and tokenizer to a local directory.

        Args:
            save_directory (str): The folder path to save into.
                                  Will be created if it doesn't exist.
        """
        os.makedirs(save_directory, exist_ok=True)

        try:
            self.pipe.model.save_pretrained(save_directory)
            self.pipe.tokenizer.save_pretrained(save_directory)
            print(f"Pipeline model and tokenizer saved to '{save_directory}'")
        except Exception as e:
            raise RuntimeError(
                f"Failed to save pipeline to '{save_directory}'.\n"
                f"Original error: {e}"
            ) from e

    @classmethod
    def load(cls, save_directory, task):
        """
        Load a previously saved pipeline from a local directory.

        Args:
            save_directory (str): The folder containing the saved model files.
            task (str): The pipeline task (must match what was used during save).

        Returns:
            ModelPipeline: A new pipeline instance loaded from disk.

        Raises:
            FileNotFoundError: If the directory does not exist.
        """
        if not os.path.isdir(save_directory):
            raise FileNotFoundError(
                f"Directory '{save_directory}' does not exist. "
                f"Did you run .save() first?"
            )

        return cls(model_name_or_path=save_directory, task=task)