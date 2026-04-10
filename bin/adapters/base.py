#!/usr/bin/env python3
"""
Base Adapter Interface
"""
import abc

class MemorySystemAdapter(abc.ABC):
    @abc.abstractmethod
    def retrieve(self, keywords, target_id=None, budget_tokens=5000):
        pass