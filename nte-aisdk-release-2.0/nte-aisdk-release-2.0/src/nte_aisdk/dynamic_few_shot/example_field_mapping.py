class DynamicFewShotExampleFieldMapping:
    input: str
    output: str

    def __init__(self, input: str, output: str, **kwargs: str):
        self.input = input
        self.output = output
        self._extra_fields = {}
        reserved_fields = {"input", "output"}

        for key, value in kwargs.items():
            if key not in reserved_fields:
                setattr(self, key, value)
                self._extra_fields[key] = value

    def get_supplementary_inputs(self) -> dict[str, str]:
        return self._extra_fields

class DynamicFewShotPromptField:
    input: str
    output: str

    def __init__(self, input: str, output: str, **kwargs: str):
        self.input = input
        self.output = output
        self._extra_fields = {}
        reserved_fields = {"input", "output"}

        for key, value in kwargs.items():
            if key not in reserved_fields:
                setattr(self, key, value)
                self._extra_fields[key] = value

    def get_supplementary_inputs(self) -> dict[str, str]:
        return self._extra_fields