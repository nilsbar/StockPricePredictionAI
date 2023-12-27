from utils.pipeline_node import PipelineNode

class Pipeline:
    def __init__(self, steps: list[PipelineNode]):
        self.steps = steps

    def run(self):
        result = None
        for step in self.steps:
            result = step.process(input = result)

        return result
