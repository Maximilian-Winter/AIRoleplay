from functools import wraps


class CommandRegistry:
    def __init__(self):
        self.commands = {}

    def register(self, name, func):
        self.commands[name.lower()] = func

    def get_command(self, name):
        if name in self.commands:
            return self.commands[name]
        else:
            return None


def chat_command(name, command_registry):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        command_registry.register(name, wrapper)
        return wrapper

    return decorator
