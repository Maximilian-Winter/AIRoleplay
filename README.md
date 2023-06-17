# AIRoleplay
Little AI roleplay program

This project implements a persona simulator using large language models (LLMs). The simulator allows a user to interact with an AI character that simulates the persona of a famous individual or a fictional character, in this case, theoretical physicist Richard Feynman.

Dependencies
This project relies on the following libraries:

llama_cpp
Please ensure these are installed before running the code.

Key Classes and Functions
## LlamaSettings: This class contains settings and hyperparameters for the Llama language models. These include model paths, thread numbers, batch sizes, token sizes, etc.

## Llama: This is the primary language model used for generating the AI character's responses. The settings for this model are specified using an instance of LlamaSettings.

## main_generate_function: This function is used to generate responses for a given prompt using the main Llama model. It takes several parameters, including the prompt, maximum number of tokens to generate, temperature (for controlling randomness), top_k and top_p (for controlling sampling), repeat penalty, and stream.

## summarizer_generate_function: This function uses a second Llama model (the 'summarizer' model) to generate summarized responses. Its parameters are similar to the main_generate_function.

## AICharacter: This class represents the AI character that the user interacts with. An AICharacter object is created with various attributes, like the character's name, description, emotional state, objectives, etc. It also includes templates for simulating conversations and generating summaries.

## main(): This function initializes the chat with the AI character and enters a loop where the user can input messages and receive responses from the AI character. The chat history is saved after each response.

## Usage
Once the required libraries are installed, the code can be run directly in a Python environment.

After launching the script, you will start a conversation with an AI that adopts the personality of Richard Feynman. You can type your messages into the terminal, and the AI will respond as if it were Feynman.

At any time, you can stop the conversation by pressing Ctrl+C.

## Customization
You can change the AI character by modifying the AICharacter object and its attributes. For example, you can change the character's name, description, objectives, or conversation templates.

Similarly, you can adjust the language models and their settings by modifying the LlamaSettings and Llama objects.

## Note
The summarizer_model instantiation is commented out in the provided code. Uncomment it if you need summarization functionality in your simulation.

## Conclusion
This persona simulator provides a simple yet powerful tool for interactive roleplay with AI characters. By leveraging the power of large language models, you can simulate conversations with a variety of personas, both real and fictional. Enjoy your conversation with Richard Feynman!
