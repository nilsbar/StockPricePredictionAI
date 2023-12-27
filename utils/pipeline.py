
class Pipeline:
    def __init__(self, steps):
        self.steps = steps

    def run(self):
        result = None
        for step in self.steps:
            result = step.process(processed_data, y_train)

        return result
