import json
import os.path
import re
from typing import List
import numpy as np
from InstructorEmbedding import INSTRUCTOR

from .memory import MemoryStream
from .prompter import Prompter
from .commands import CommandRegistry
from .chat_history import ChatTurn


def load_personality(filename):
    # Initialize the variables
    description = None
    scenario = None
    feelings = None
    goals = None
    save_directory = None
    location = None
    username = None
    character_name = None

    # Open the file and read it line by line
    with open(filename, 'r') as file:
        lines = file.readlines()

    for line in lines:
        # Remove any leading/trailing whitespace
        line = line.strip()

        # Check if the line matches the format for each piece of information
        if line.startswith("Description:"):
            description = line[len("Description:"):].strip()
        elif line.startswith("Scenario:"):
            scenario = line[len("Scenario:"):].strip()
        elif line.startswith("Feelings:"):
            feelings = line[len("Feelings:"):].strip()
        elif line.startswith("Goals:"):
            # Split the string into a list of goals, removing leading/trailing whitespace
            goals = [goal.strip() for goal in line[len("Goals:"):].split(',')]
        elif line.startswith("SaveDirectory:"):
            save_directory = line[len("SaveDirectory:"):].strip()
        elif line.startswith("Location:"):
            location = line[len("Location:"):].strip()
        elif line.startswith("Username:"):
            username = line[len("Username:"):].strip()
        elif line.startswith("CharacterName:"):
            character_name = line[len("CharacterName:"):].strip()

    return description, scenario, feelings, goals, save_directory, location, username, character_name


class AICharacter:
    def __init__(self, main_generate_function, summarizer_generate_function, tokenizer_encode_function,
                 character_name="Grey Fox", user_name="User",
                 system_message="You are a the user's personal assistant.", objectives=None, max_output_length=350,
                 chat_template_filename="", rate_memory_importance_template_filename="", summarizer_template_filename="",
                 command_registry=None,
                 save_dir="./pa_data",
                 character_description="",
                 emotional_state="",
                 location="",
                 scenario="",
                 max_context_size=1024,
                 generation_parameters=None, manual_summarize=False, debug_output=False):

        self.update_counter = 0
        self.max_context_size = max_context_size
        self.rate_memory_prompt = Prompter.from_string(rate_memory_importance_template_filename)
        self.manual_summarize = manual_summarize
        self.tokenizer_encode_function = tokenizer_encode_function
        self.max_output_length = max_output_length
        if generation_parameters is None:
            self.generation_parameters = {}
        else:
            self.generation_parameters = generation_parameters

        if command_registry is None:
            self.command_registry = CommandRegistry()
        else:
            self.command_registry = command_registry
        self.character_description = character_description
        if objectives is None:
            self.objectives = ["Be helpful, friendly and honest to the user!"]
        else:
            self.objectives = objectives
        self.assistant_name = character_name
        self.objectives_list = "\n".join(self.objectives)
        self.save_dir = save_dir
        self.memory_stream = MemoryStream(self.create_embeddings)
        self.chat_history = []

        self.user_name = user_name
        self.system_message = system_message

        self.prompt_summary = Prompter.from_string(summarizer_template_filename)

        self.chat_prompt = Prompter.from_string(chat_template_filename)
        self.memorize_summaries_interval = 4
        self.dialogue_summaries_count = 0
        self.current_summaries = []
        self.additional_msgs_in_context = 2
        self.debug_output = debug_output
        self.main_generate_function = main_generate_function
        self.summarizer_generate_function = summarizer_generate_function
        self.embedding_model = INSTRUCTOR('hkunlp/instructor-large')
        self.emotional_state = emotional_state
        self.location = location
        self.scenario = scenario
        if not os.path.exists(save_dir):
            self.save_bot()
        else:
            self.load_bot()

    def init_chat(self):
        for chat in self.chat_history:
            print(chat)

    def save_bot(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.memory_stream.save_to_json(f"{self.save_dir}/memory_stream.json")
        self.save_chat_history()
        bot_settings = [self.assistant_name, self.user_name, self.objectives, self.scenario, self.location,
                        self.emotional_state, self.character_description, self.update_counter]
        with open(f"{self.save_dir}/bot_settings.json", 'w') as file:
            json.dump(bot_settings, file)

    def load_bot(self):
        if os.path.exists(self.save_dir):
            self.memory_stream.load_from_json(f"{self.save_dir}/memory_stream.json")
            self.load_chat_history()
            with open(f"{self.save_dir}/bot_settings.json", 'r') as file:
                bot_settings = json.load(file)
                self.assistant_name = bot_settings[0]
                self.user_name = bot_settings[1]
                self.objectives = bot_settings[2]
                self.objectives_list = "\n".join(self.objectives)
                self.scenario = bot_settings[3]
                self.location = bot_settings[4]
                self.emotional_state = bot_settings[5]
                self.character_description = bot_settings[6]
                self.update_counter = bot_settings[7]

    def create_embeddings(self, sentences: List[str]) -> List[np.ndarray]:
        """Create embeddings for a list of sentences."""
        embeddings = self.embedding_model.encode(sentences)
        # Convert the embeddings to float16 for efficient memory utilization
        return [np.array(embed) for embed in embeddings]

    def save_chat_history(self):
        chat_history_dict = [chat_turn.to_dict() for chat_turn in self.chat_history]
        with open(f"{self.save_dir}/chat_history.json", 'w') as file:
            json.dump(chat_history_dict, file)

    def load_chat_history(self):
        try:
            with open(f"{self.save_dir}/chat_history.json", 'r') as file:
                chat_history_dict = json.load(file)
                self.chat_history = [ChatTurn.chat_turn_from_dict(chat_turn_dict) for chat_turn_dict in
                                     chat_history_dict]
        except FileNotFoundError:
            print(f"File '{self.save_dir}/chat_history.json' not found.")
            self.chat_history = []

    def summarize_chat_history_manual(self):
        mem_history = ""
        k = input("Number of Turns to Summarize: ")
        mem = self.chat_history[: int(k)]
        for i in range(len(mem)):
            mem_history += mem[i] + "\n"
            self.chat_history.remove(mem[i])

        print(mem)

        summary = input("Summary: ")
        self.memory_stream.add_memory(summary)
        self.memory_stream.save_to_json(f"{self.save_dir}/memory_stream.json")

    def memorize_chat_history(self, k=5):
        mem_history = ""
        mem = self.chat_history[: k]
        for chat in mem:
            mem_history += f"{chat.query_message.owner}: {chat.query_message.message}\n{chat.response_message.owner}:{chat.response_message.message}\n"
            chat.is_in_memory = True

        summary_prompt = self.prompt_summary.generate_prompt(
            {"conversation_history": mem_history, "assistant_name": self.assistant_name})
        output_summary = self.summarizer_generate_function(summary_prompt)
        if self.debug_output:
            print(output_summary)
        self.current_summaries.append(output_summary)

        rate_prompt = self.rate_memory_prompt.generate_prompt(
            {"memory": output_summary})
        output_rating = self.summarizer_generate_function(rate_prompt)
        match = re.search(r'\d+', output_rating)
        if match:
            importance = int(match.group())
        else:
            importance = 1

        self.memory_stream.add_memory(output_summary, importance=importance)
        self.memory_stream.save_to_json(f"{self.save_dir}/memory_stream.json")
        self.save_chat_history()
        self.dialogue_summaries_count += 1

    def create_contextual_query(self, user_input):
        mem_history = f"{self.assistant_name}'s feelings: {self.emotional_state}\n{self.assistant_name}'s location: {self.location}\n"
        if len(self.chat_history) > 0:
            chat = self.chat_history[-1]
            mem_history += f"{chat.query_message.message}\n{chat.response_message.message}\n{user_input}"
        return mem_history

    def build_conversation_prompt(self, user_input):
        additional_context = ""
        history = ""
        query_memories = self.create_contextual_query(user_input)
        memories = self.memory_stream.retrieve_memories(query_memories, self.additional_msgs_in_context)
        memories.sort(key=lambda memory: memory.creation_timestamp)
        for m in memories:
            additional_context += self.remove_empty_lines(m.description) + "\n"

        for chat in self.chat_history:
            if not chat.is_in_memory:
                history += f"{chat.query_message.owner}: {chat.query_message.message}\n{chat.response_message.owner}:{chat.response_message.message}\n"

        history = history[:-1]
        prompt_str = self.chat_prompt.generate_prompt(
            {"assistant_name": self.assistant_name, "location": self.location,
             "user_name": self.user_name, "history": history, "emotional_state": self.emotional_state,
             "scenario": self.scenario, "character": self.character_description,
             "input": user_input, "system_message": self.system_message,
             "additional_context": additional_context, "objectives": self.objectives_list})

        return prompt_str

    def remove_empty_lines(self, string):
        lines = string.splitlines()
        non_empty_lines = [line for line in lines if line.strip() != ""]
        string_no_empty_lines = "\n".join(non_empty_lines)
        return string_no_empty_lines

    def conversation(self, user_input):
        if user_input == "":
            return
        command_result, has_found_command = self.process_commands(user_input)
        if not has_found_command:
            if not self.debug_output:
                print(f"User: {user_input}\nModel:", end="")

            prompt_str = self.build_conversation_prompt(user_input)
            prompt_length = len(self.tokenizer_encode_function(prompt_str))
            summarize_count = 0
            while True:
                if (prompt_length + self.max_output_length) >= self.max_context_size:
                    if self.manual_summarize:
                        self.summarize_chat_history_manual()
                    else:
                        self.memorize_chat_history(4)
                        # if self.dialogue_summaries_count == self.memorize_summaries_interval:
                        #    self.summarize_summaries()
                        summarize_count += 1
                    prompt_str = self.build_conversation_prompt(user_input)
                    prompt_length = len(self.tokenizer_encode_function(prompt_str))
                else:
                    break

            output = self.main_generate_function(prompt_str)
            turn = ChatTurn(self.user_name, self.remove_empty_lines(user_input), self.assistant_name,
                            self.remove_empty_lines(output))
            self.chat_history.append(turn)
            self.save_chat_history()
            self.update_counter += 1
            if self.update_counter == 5:
                # output = self.summarizer_generate_function(self.build_emotion_prompt())
                # self.emotional_state = self.remove_empty_lines(output)
                # output = self.summarizer_generate_function(self.build_objectives_prompt())
                # self.objectives_list = self.remove_empty_lines(output)
                self.update_counter = 0
            self.save_bot()

    def process_commands(self, input_text):
        command_pattern = re.compile(r"@(\w+)(.*)")
        match = command_pattern.match(input_text.strip())
        has_found_command = False
        if match:
            command_name, args_str = match.groups()
            args = [arg.strip() for arg in args_str.split() if arg.strip()]
            command_method = self.command_registry.get_command(command_name)
            has_found_command = True
            if command_method:
                return command_method(self, *args), has_found_command
        return None, has_found_command
