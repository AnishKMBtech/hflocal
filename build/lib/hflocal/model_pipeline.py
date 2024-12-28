from transformers import pipeline

class ModelPipeline:
    def __init__(self, model_name_or_path, task):
        """
        Initialize the model pipeline.

        Args:
            model_name_or_path (str): The name or path of the pre-trained model.
            task (str): The task for the pipeline (e.g., 'fill-mask', 'sentiment-analysis').
        """
        self.pipeline = pipeline(task, model=model_name_or_path)

    def __call__(self, *args, **kwargs):
        """
        Call the pipeline with the given arguments.

        Args:
            *args: Positional arguments for the pipeline.
            **kwargs: Keyword arguments for the pipeline.

        Returns:
            The result of the pipeline.
        """
        return self.pipeline(*args, **kwargs)