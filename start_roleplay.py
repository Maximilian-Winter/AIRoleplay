from llama_cpp import Llama
from ai_roleplay import AICharacter, load_personality
from ai_roleplay.default_commands import command_registry


class LlamaSettings:
    model: str = ""
    n_ctx: int = 4096
    n_batch: int = 128
    n_threads: int = 4
    f16_kv: bool = True
    use_mlock: bool = True
    embedding: bool = False
    last_n_tokens_size: int = 64
    n_gpu_layers: int = 0
    verbose: bool = False


settings = LlamaSettings()
settings.model = "../../ggml-v3-models/WizardLM-Uncensored-SuperCOT-Storytelling.ggmlv3.q4_0.bin"
settings.n_batch = 768
settings.n_gpu_layers = 7
settings.n_threads = 12
settings.last_n_tokens_size = 2048
settings.verbose = False
settings.embedding = False

settings_summarizer = LlamaSettings()
settings_summarizer.model = "../../ggml-v3-models/wizardLM-13B-Uncensored.ggmlv3.q6_K.bin"
settings_summarizer.n_batch = 512
settings_summarizer.n_gpu_layers = 3
settings_summarizer.n_threads = 12
settings_summarizer.last_n_tokens_size = 2048
settings_summarizer.embedding = False
settings_summarizer.verbose = False

main_model = Llama(
    settings.model,
    n_gpu_layers=settings.n_gpu_layers,
    f16_kv=settings.f16_kv,
    use_mlock=settings.use_mlock,
    embedding=settings.embedding,
    n_threads=settings.n_threads,
    n_batch=settings.n_batch,
    n_ctx=settings.n_ctx,
    last_n_tokens_size=settings.last_n_tokens_size,
    verbose=settings.verbose
)

summarizer_model = Llama(
    settings_summarizer.model,
    n_gpu_layers=settings_summarizer.n_gpu_layers,
    f16_kv=settings_summarizer.f16_kv,
    use_mlock=settings_summarizer.use_mlock,
    embedding=settings_summarizer.embedding,
    n_threads=settings_summarizer.n_threads,
    n_batch=settings_summarizer.n_batch,
    n_ctx=2048,
    last_n_tokens_size=settings_summarizer.last_n_tokens_size,
    verbose=settings_summarizer.verbose
)


def main_generate_function(prompt: str = "", max_tokens: int = 500, temperature: float = 0.7,
                           top_k: int = 0, top_p: float = 0.5, repeat_penalty: float = 1.2, stream: bool = True):
    if character.debug_output:
        print(prompt)
    result = main_model(
        f"{prompt}",
        max_tokens=max_tokens,
        stream=stream,
        stop=['```python', 'Input:', 'Response:', f'{character.user_name}:', '</conversation>', '###',
              'Additional context:'],
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        mirostat_mode=0,
        mirostat_tau=5.0,
        mirostat_eta=0.1,
        repeat_penalty=repeat_penalty,
        tfs_z=0.97
    )
    if character.debug_output:
        print(result)
    output = ""
    token_generator = result
    for out in token_generator:
        output += out['choices'][0]['text']
        print(out['choices'][0]['text'], end="", flush=True)
    print("")
    return output


def summarizer_generate_function(prompt: str = "", max_tokens: int = 200, temperature: float = 0.1,
                                 top_k: int = 40, top_p: float = 0.9, repeat_penalty: float = 1.2,
                                 stream: bool = True):
    if character.debug_output:
        print(prompt)
    result = summarizer_model(
        f"{prompt}",
        max_tokens=max_tokens,
        stream=stream,
        stop=['User:', f'{character.user_name}:', '</conversation>', '###', 'Additional context:'],
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        mirostat_mode=0,
        mirostat_tau=5.0,
        mirostat_eta=0.1,
        repeat_penalty=repeat_penalty,
    )
    if character.debug_output:
        print(result)
    output = ""
    token_generator = result
    for out in token_generator:
        output += out['choices'][0]['text']
        print(out['choices'][0]['text'], end="", flush=True)
    print("")
    return output


description, scenario, feelings, goals, save_directory, location, username, character_name = load_personality('Personalities/RichardFeynman.txt')
character = AICharacter(main_generate_function=main_generate_function,
                        summarizer_generate_function=summarizer_generate_function,
                        tokenizer_encode_function=main_model.tokenizer().encode,
                        character_name=character_name, user_name=username,
                        system_message="Adopt the personality described in the character section from the user and respond to the user's last message in the conversation history. Consider the user provided scenario, location, character's feelings, character's goals, character's memories and conversation history, when writing a response. Ensure that the response is coherent and in character.",
                        scenario=scenario,
                        location=location,
                        emotional_state=feelings,
                        save_dir=save_directory,
                        character_description=description,
                        objectives=goals,
                        chat_template_filename="PromptTemplates/chat_instruction.txt",
                        rate_memory_importance_template_filename="PromptTemplates/rate_memory.txt",
                        summarizer_template_filename="PromptTemplates/summary.txt",
                        command_registry=command_registry,
                        max_output_length=500, max_context_size=settings.n_ctx, manual_summarize=False,
                        debug_output=True)

character.init_chat()

while True:
    user_input = input(">")
    character.conversation(user_input)
    character.save_bot()
