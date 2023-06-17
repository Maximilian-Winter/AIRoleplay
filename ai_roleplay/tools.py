from functools import wraps

from googlesearch import search


def split_string_into_chunks(input_string, k):
    if k <= 0:
        raise ValueError("k must be a positive integer")

    chunks = []
    for i in range(0, len(input_string), k):
        chunk = input_string[i:i + k]
        chunks.append(chunk)

    return chunks


def extract_text_from_url_tool(input_text):
    url = input_text.strip()
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return f"Error: Unable to fetch URL ({url}). Error details: {e}"

    soup = BeautifulSoup(response.content, "html.parser")
    for script in soup(["script", "style"]):
        script.decompose()
    text = soup.get_text()
    return text.strip()


class Tool:
    def __init__(self, name, function, description, usage_example):
        self.name = name
        self.function = function
        self.description = description
        self.usage_example = usage_example


class ToolRegistry:
    def __init__(self):
        self.tools = {}

    def register_tool(self, tool):
        if tool.name in self.tools:
            raise ValueError(f"A tool with the name '{tool.name}' already exists.")
        self.tools[tool.name] = tool

    def get_tool(self, name):
        if name not in self.tools:
            raise KeyError(f"No tool with the name '{name}' found.")
        return self.tools[name].function

    def get_overview(self):
        overview = ""
        for tool_name, tool in self.tools.items():
            overview += f"Tool Name: {tool_name} Tool Description: {tool.description} Tool Usage Example: {tool.usage_example}\n"

        return overview

    def parse_string_for_tool(self, string):
        words = string.split()  # split string into words
        if words:
            tool_name = words[0]
            if tool_name in self.tools:
                args = words[1:]  # all words after the tool name are treated as arguments
                return tool_name, args

        # return None or raise an exception if no valid tool command is found
        return None, None


def agent_tool(tool_registry, name, description, usage_example):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        tool_registry.register_tool(Tool(name, wrapper, description, usage_example))
        return wrapper

    return decorator


@agent_tool(None, "google_search", "Search for information on the web.",
            "google_search 'Spaghetti with Meatballs recipe'")
def google_search(personal_assistant, query, num_results=3):
    results = []
    counter = 1
    for j in search(query, advanced=True, num_results=num_results):
        title = j.title
        description = j.description
        url = j.url

        formatted_result = f"{counter}. Title: {title}\nDescription: {description}\nURL: {url}\n"
        print(formatted_result)
        counter += 1
        text = extract_text_from_url_tool(url)
        print(text)
        chunks = split_string_into_chunks(text, 512)

        for chunk in chunks:
            personal_assistant.memory_stream.add_memory(chunk)

    memories = personal_assistant.memory_stream.retrieve_memories(query, 5, alpha_recency=0, alpha_importance=0)

    print("Query:", query)
    for j in memories:
        results.append(j)
        print("Result:", j)

    result_string = ''.join(results)
    print("ResultString:", result_string)
    return result_string
