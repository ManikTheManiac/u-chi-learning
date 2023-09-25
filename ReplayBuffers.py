from collections import deque
import random
import numpy as np
import torch

class Memory(object):
    def __init__(self, memory_size: int, device='cpu') -> None:
        self.memory_size = memory_size
        self.device = device
        self.buffer = deque(maxlen=self.memory_size)

    def add(self, experience) -> None:
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size: int, continuous: bool = True):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if continuous:
            rand = random.randint(0, len(self.buffer) - batch_size)
            batch = [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            # Sample according to priority:
            # dist = np.array([1 / (i + 1) for i in range(len(self.buffer))])
            # dist = dist / np.sum(dist)
            indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)#, p=dist)
            batch = [self.buffer[i] for i in indexes]
        
        states, next_states, actions, next_actions, rewards, dones = zip(*batch)
        # Now convert to tensors:
        states = torch.from_numpy(np.array(states, dtype=np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states, dtype=np.float32)).to(self.device)
        actions = torch.from_numpy(np.array(actions, dtype=np.float32)).to(self.device)
        next_actions = torch.from_numpy(np.array(next_actions, dtype=np.float32)).to(self.device)
        rewards = torch.from_numpy(np.array(rewards, dtype=np.float32)).to(self.device)
        dones = torch.from_numpy(np.array(dones, dtype=np.float32)).to(self.device)

        return states, next_states, actions, next_actions, rewards, dones

    def clear(self):
        self.buffer.clear()
