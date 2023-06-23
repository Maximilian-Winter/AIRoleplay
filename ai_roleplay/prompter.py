import re


class Prompter:
    def __init__(self, template_file=None, template_string=None):
        if template_file:
            with open(template_file, "r") as file:
                self.template = file.read()
        elif template_string:
            self.template = template_string
        else:
            raise ValueError("Either 'template_file' or 'template_string' must be provided")

    @classmethod
    def from_string(cls, template_string):
        return cls(template_string=template_string)

    @classmethod
    def from_file(cls, template_file):
        with open(template_file, "r") as file:
            template_string = file.read()
        return cls(template_string=template_string)

    def generate_prompt(self, template_fields):
        def replace_placeholder(match):
            placeholder = match.group(1)
            return template_fields.get(placeholder, match.group(0))

        prompt = re.sub(r"\{(\w+)\}", replace_placeholder, self.template)
        return prompt
