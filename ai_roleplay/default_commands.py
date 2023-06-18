from . import AICharacter, CommandRegistry, chat_command

command_registry = CommandRegistry()

@chat_command("set_location", command_registry)
def set_location(personal_assistant: AICharacter):
    user_input = input("Location:")
    personal_assistant.location = user_input

@chat_command("set_scenario", command_registry)
def set_scenario(personal_assistant: AICharacter):
    user_input = input("Scenario:")
    personal_assistant.scenario = user_input

@chat_command("set_emotional_state", command_registry)
def set_emotional_state(personal_assistant: AICharacter):
    user_input = input("Emotional State:")
    personal_assistant.emotional_state = user_input


@chat_command("set_objectives", command_registry)
def set_objectives(personal_assistant: AICharacter):
    objectives = []
    user_input = input("How many objectives:")
    for i in range(int(user_input)):
        objective = input("Objective:")
        objectives.append(objective)
    personal_assistant.objectives_list = "\n".join(objectives)
    personal_assistant.objectives = objectives

@chat_command("add_memory", command_registry)
def add_memory(personal_assistant: AICharacter):
    user_input = input("Memory:")
    personal_assistant.memory_stream.add_memory(user_input)

@chat_command("update_all_embeddings", command_registry)
def update_all_embeddings(personal_assistant: AICharacter):
    personal_assistant.memory_stream.update_all_embeddings()

@chat_command("test_summarizer", command_registry)
def test_summarizer(personal_assistant: AICharacter, k, temp, top_k, top_p):
    personal_assistant.memorize_chat_history_test(int(k), float(temp), int(top_k), float(top_p))

@chat_command("test_mems", command_registry)
def test_mems(personal_assistant: AICharacter):
    personal_assistant.test_memories()

@chat_command("set_temperature", command_registry)
def set_temperature(personal_assistant: AICharacter, temp):
    personal_assistant.generation_parameters['temperature'] = float(temp)


@chat_command("remove_last_turn", command_registry)
def remove_last_turn(personal_assistant: AICharacter, count=1):
    personal_assistant.chat_history = personal_assistant.chat_history[:-int(count)]
    personal_assistant.init_chat()
    personal_assistant.save_chat_history()


@chat_command("summarize_dialogue_parts", command_registry)
def summarize_dialogue_parts(personal_assistant: AICharacter):
    personal_assistant.summarize_summaries()
    personal_assistant.memory_stream.save_to_json(f"{personal_assistant.save_dir}/memory_stream.json")




@chat_command("memorize_chat_history", command_registry)
def memorize_chat_history(personal_assistant: AICharacter, count=2):
    personal_assistant.memorize_chat_history(int(count))
    personal_assistant.init_chat()


@chat_command("edit_memory", command_registry)
def edit_memory(personal_assistant: AICharacter):
    mem_counter = 1
    for memory in personal_assistant.memory_stream.memories:
        print(f"{mem_counter}: {memory.description}")
        mem_counter += 1

    while True:
        choice = input("Enter the memory number you want to edit or delete, 'b' to go back, or 'q' to quit: ")

        if choice.lower() == 'q':
            break
        elif choice.lower() == 'b':
            return

        if not choice.isdigit() or int(choice) < 1 or int(choice) > len(personal_assistant.memory_stream.memories):
            print("Invalid memory number. Please enter a valid number.")
            continue

        memory_index = int(choice) - 1  # Convert to zero-based index
        memory = personal_assistant.memory_stream.memories[memory_index]
        action = input(
            f"You selected memory '{memory.description}'. Do you want to (e)dit, (d)elete, or (b)ack? ").lower()

        if action == 'e':
            new_description = input("Enter the new description: ")
            memory.description = new_description
            print(f"Memory '{choice}' has been updated.")
        elif action == 'd':
            personal_assistant.memory_stream.remove_memory(memory.description)
            print(f"Memory '{choice}' has been deleted.")
        elif action == 'b':
            continue
        else:
            print("Invalid option. Please choose to (e)dit, (d)elete, or go (b)ack.")


@chat_command("edit_chat_history", command_registry)
def edit_chat_history(personal_assistant: AICharacter):
    turn_counter = 1
    for chat_turn in personal_assistant.chat_history:
        print(f"{turn_counter}: {chat_turn}")
        turn_counter += 1

    while True:
        choice = input("Enter the turn number you want to delete or edit, 'b' to go back, or 'q' to quit: ")

        if choice.lower() == 'q':
            break
        elif choice.lower() == 'b':
            return

        if not choice.isdigit() or int(choice) < 1 or int(choice) > len(personal_assistant.chat_history):
            print("Invalid turn number. Please enter a valid number.")
            continue

        turn_index = int(choice) - 1  # Convert to zero-based index
        chat_turn = personal_assistant.chat_history[turn_index]
        action = input(f"You selected turn '{chat_turn}'. Do you want to (e)dit, (d)elete, or (b)ack? ").lower()

        if action == 'd':
            personal_assistant.chat_history.pop(turn_index)
            print(f"Turn '{choice}' has been deleted.")
        elif action == 'e':
            # Separate the participant's name from the message
            participant, message = chat_turn.split(': ', 1)
            if participant != personal_assistant.user_name and participant != personal_assistant.assistant_name:
                print("Invalid turn. Unable to edit.")
                continue

            new_message = input(f"Enter the new message for '{participant}': ")
            personal_assistant.chat_history[turn_index] = f"{participant}: {new_message}"
            print(f"Turn '{choice}' has been updated.")
        elif action == 'b':
            continue
        else:
            print("Invalid option. Please choose to (e)dit, (d)elete, or go (b)ack.")