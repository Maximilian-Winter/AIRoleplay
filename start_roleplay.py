from llama_cpp import Llama
from ai_roleplay import AICharacter
from ai_roleplay.default_commands import command_registry

class LlamaSettings:
    model: str = ""
    n_ctx: int = 2048
    n_batch: int = 128
    n_threads: int = 4
    f16_kv: bool = True
    use_mlock: bool = True
    embedding: bool = False
    last_n_tokens_size: int = 64
    n_gpu_layers: int = 0
    verbose: bool = False

settings = LlamaSettings()
settings.model = "../../ggml-v3-models/Dans-PersonalityEngine-30b-ggml-q5_0.bin"
settings.n_batch = 512
settings.n_gpu_layers = 7
settings.n_threads = 12
settings.last_n_tokens_size = 2048
settings.verbose = False

settings_summarizer = LlamaSettings()
settings_summarizer.model = "../../ggml-v3-models/WizardLM-7B-uncensored.ggmlv3.q5_1.bin"
settings_summarizer.n_batch = 512
settings_summarizer.n_gpu_layers = 4
settings_summarizer.n_threads = 12
settings_summarizer.last_n_tokens_size = 2048
settings_summarizer.embedding = True
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

#summarizer_model = Llama(
#    settings_summarizer.model,
#    n_gpu_layers=settings_summarizer.n_gpu_layers,
#    f16_kv=settings_summarizer.f16_kv,
#    use_mlock=settings_summarizer.use_mlock,
#    embedding=settings_summarizer.embedding,
#    n_threads=settings_summarizer.n_threads,
#    n_batch=settings_summarizer.n_batch,
#    n_ctx=settings_summarizer.n_ctx,
#    last_n_tokens_size=settings_summarizer.last_n_tokens_size,
#    verbose=settings_summarizer.verbose
#)


def main_generate_function(prompt: str = "", max_tokens: int = 500, temperature: float = 0.3,
                           top_k: int = 40, top_p: float = 0.75, repeat_penalty: float = 1.2, stream: bool = True):
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


def summarizer_generate_function(prompt: str = "", max_tokens: int = 200, temperature: float = 0.0001,
                                 top_k: int = 0, top_p: float = 1, repeat_penalty: float = 1.2,
                                 stream: bool = False):
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
    return result['choices'][0]['text']


template_summary = """### Instruction:
Generate a short summary of the following partial dialogue as a long term memory! Pay close attention to not mix up any of the details! Don't make assumptions!
### Input:
Dialogue:
{history}
### Response:
Summary:"""

template_summary_parts = """### Instruction:
Generate a short summary of the following partial memories as a long term memory! Pay close attention to not mix up any of the details! Don't make assumptions!
### Input:
Memories:
{history}
### Response:
Summary:"""

chat_template_guanaco = """### Instruction:
{history}
### Input:
System: {system_message}
{assistant_name}'s objectives:
{objectives}
Additional Context:
{additional_context}
Current Context:
{user_name}: {input}


### Response:
{assistant_name}:"""

chat_template_instruction = """### Instruction:
{system_message}

Character:
{character}

Scenario:
{scenario}

{assistant_name}'s Emotional State:
{emotional_state}

{assistant_name}'s Goals:
{objectives}

Current Location:
{location}

Additional Context:
{additional_context}
Conversation History:
{history}
### Input:
{user_name}: {input}


### Response:
{assistant_name}:"""


character = """Richard Feynman, the charismatic and influential theoretical physicist renowned for his work in quantum mechanics, quantum electrodynamics, and particle physics."""
character = AICharacter(main_generate_function=main_generate_function,
                        summarizer_generate_function=summarizer_generate_function,
                        tokenizer_encode_function=main_model.tokenizer().encode,
                        embedding_function=main_model.embed,
                        character_name="Richard Feynman", user_name="Maximilian Winter",
                        system_message="Adopt the personality described in the character section below and respond to the current user message in input. Consider the complete conversation history, the additional context, the character's current location, situation, emotional state and goals below when writing a response. Ensure that the response is coherent, relevant, and concise. Avoid including any information that is not based on the given context. Be creative and engaging in your response.",
                        scenario="It's 1965, and Feynman has just won the Nobel Prize in Physics, bringing with it a wave of attention and expectations.",
                        location="At work, in her office.",
                        emotional_state="Feynman is feeling excited and gratified, but also slightly overwhelmed by the sudden surge in public interest and the expectations for his future work.",
                        save_dir="pa_data_rf",
                        character_description=character,
                        objectives=["His main goal is to balance his newfound fame with his desire to continue deep, focused work in theoretical physics. He also wants to use his platform to inspire a new generation of scientists without getting too caught up in the celebrity status."],
                        chat_template=chat_template_instruction,
                        summarizer_template=template_summary,
                        summarizer_summaries_template=template_summary_parts,
                        command_registry=command_registry,
                        max_output_length=500, manual_summarize=True, debug_output=True)

character.init_chat()

while True:
    user_input = input(">")
    character.conversation(user_input)
    character.save_bot()
