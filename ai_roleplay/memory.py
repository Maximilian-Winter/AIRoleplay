import datetime
import json
from copy import copy
from typing import List
import numpy as np
from dataclasses import dataclass

from scipy.spatial.distance import cosine


@dataclass
class Memory:
    """Class representing a memory with a description, timestamps, importance, and an embedding."""
    description: str
    creation_timestamp: datetime.datetime
    last_access_timestamp: datetime.datetime
    embedding: np.ndarray


class MemoryStream:
    """
    Class representing a stream of memories. It has methods to add memories, compute recency and relevance,
    retrieve memories, and perform other operations related to memories.
    """

    def __init__(self, get_embedding):
        self.memories: List[Memory] = []
        self.get_embedding = get_embedding

    def add_memory(self, description: str, date: datetime.datetime = datetime.datetime.now(), importance: float = 1.0):
        embedding = self.get_embedding([description])[0]  # Convert to batch
        embedding = np.array(embedding)  # Reduce precision
        memory = Memory(description, date, date, embedding)
        self.memories.append(memory)

    def update_all_embeddings(self):
        for memory in self.memories:
            embedding = self.get_embedding([memory.description])[0]  # Convert to batch
            embedding = np.array(embedding)
            memory.embedding = embedding

    def update_last_access(self, memory, date):
        memory.last_access_timestamp = date

    def remove_memory(self, description: str):
        """Remove a memory with a given description from the memory stream."""
        self.memories = [memory for memory in self.memories if memory.description != description]
    def compute_recency(self, memory, date):
        decay_factor = 0.99
        time_diff = date - memory.last_access_timestamp
        hours_diff = time_diff.total_seconds() / 3600
        recency = decay_factor ** hours_diff
        return recency

    def compute_relevance(self, memory_embedding, query_embedding):
        relevance = 1 - cosine(memory_embedding, query_embedding)
        return relevance

    def compute_scores(self, query_embedding, date, alpha_recency=1, alpha_relevance=1):
        scores = []
        for memory in self.memories:
            recency = self.compute_recency(memory, date)
            relevance = self.compute_relevance(memory.embedding, query_embedding)
            score = alpha_recency * recency + alpha_relevance * relevance
            scores.append(score)
        return np.array(scores, dtype=np.float16)

    def normalize_scores(self, scores):
        min_score, max_score = np.min(scores), np.max(scores)
        if min_score == max_score:
            return np.zeros_like(scores)
        return (scores - min_score) / (max_score - min_score)

    def get_top_indices(self, scores, k):
        return scores.argsort()[-k:][::-1]

    def retrieve_memories(self, query, k, date=datetime.datetime.now(), alpha_recency=1,
                          alpha_relevance=1):
        if len(self.memories) > 0:
            query_embedding = self.get_embedding([query])[0]  # Convert to batch
            query_embedding = np.array(query_embedding)
            scores = self.compute_scores(query_embedding, date, alpha_recency, alpha_relevance)
            normalized_scores = self.normalize_scores(scores)
            top_indices = self.get_top_indices(normalized_scores, k)
            retrieved_memories = [self.memories[i] for i in top_indices]
            for memory in retrieved_memories:
                self.update_last_access(memory, date)
            return retrieved_memories
        else:
            return []

    def get_last_k_memories(self, k, timestamp=datetime.datetime.now()):
        memories_before_timestamp = [memory for memory in self.memories if memory.creation_timestamp <= timestamp]
        sorted_memories = sorted(memories_before_timestamp, key=lambda memory: memory.creation_timestamp, reverse=True)
        last_k_memories = sorted_memories[:k]
        return last_k_memories

    def save_to_json(self, filename):
        serialized_memories = []
        for memory in self.memories:
            memory_dict = copy(memory.__dict__)  # Convert namedtuple to dictionary
            memory_dict['creation_timestamp'] = memory_dict[
                'creation_timestamp'].isoformat()  # Convert datetime to string
            memory_dict['last_access_timestamp'] = memory_dict['last_access_timestamp'].isoformat()
            memory_dict['embedding'] = memory_dict['embedding'].tolist()  # Convert numpy array to list
            serialized_memories.append(memory_dict)

        with open(filename, 'w') as f:
            json.dump(serialized_memories, f)

    def load_from_json(self, filename):
        with open(filename, 'r') as f:
            serialized_memories = json.load(f)

        self.memories = []
        for memory_dict in serialized_memories:
            memory_dict['creation_timestamp'] = datetime.datetime.fromisoformat(
                memory_dict['creation_timestamp'])  # Convert string to datetime
            memory_dict['last_access_timestamp'] = datetime.datetime.fromisoformat(memory_dict['last_access_timestamp'])
            memory_dict['embedding'] = np.array(memory_dict['embedding'],
                                                dtype=np.float16)  # Convert list to numpy array
            memory = Memory(**memory_dict)
            self.memories.append(memory)
