from dataclasses import dataclass, asdict


@dataclass
class ChatMessage:
    owner: str
    message: str

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def chat_message_from_dict(chat_message_dict):
        return ChatMessage(
            chat_message_dict['owner'],
            chat_message_dict['message']
        )


class ChatTurn:

    def __init__(self, query_owner, query_message, respond_owner, respond_message):
        self.query_message = ChatMessage(query_owner, query_message)
        self.response_message = ChatMessage(respond_owner, respond_message)
        self.is_in_memory = False

    def __str__(self):
        return f"{self.query_message.owner}: {self.query_message.message}\n{self.response_message.owner}: {self.response_message.message}\n"

    def to_dict(self):
        return {
            'query_message': self.query_message.to_dict(),
            'response_message': self.response_message.to_dict(),
            'is_in_memory': self.is_in_memory
        }

    @staticmethod
    def chat_turn_from_dict(chat_turn_dict):
        query_message = ChatMessage.chat_message_from_dict(chat_turn_dict['query_message'])
        response_message = ChatMessage.chat_message_from_dict(chat_turn_dict['response_message'])
        is_in_memory = chat_turn_dict['is_in_memory']

        chat_turn = ChatTurn(
            query_message.owner,
            query_message.message,
            response_message.owner,
            response_message.message
        )
        chat_turn.is_in_memory = is_in_memory
        return chat_turn