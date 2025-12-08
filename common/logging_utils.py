# Copyright 2025 Keen Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Consistent logging utilities for all agents.
Single source of truth for logging behavior.
"""

from __future__ import annotations

from typing import Optional


class AgentLogger:
    """
    Simple logger that writes to both stdout and an optional file.
    
    Provides consistent logging across all agents with minimal overhead.
    """
    
    def __init__(self, log_file: Optional[str] = None, prefix: str = ""):
        """
        Initialize the logger.
        
        Args:
            log_file: Optional path to log file. If None, only prints to stdout.
            prefix: Optional prefix to add to all log messages.
        """
        self.log_file = log_file
        self.prefix = prefix
        
        # Clear the log file if it exists
        if self.log_file:
            with open(self.log_file, "w", encoding="utf-8") as f:
                f.write("")
    
    def log(self, message: str) -> None:
        """
        Log a message to stdout and optionally to file.
        
        Args:
            message: Message to log
        """
        full_message = f"{self.prefix}{message}" if self.prefix else message
        print(full_message)
        
        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(full_message + "\n")
    
    def __call__(self, message: str) -> None:
        """Allow logger to be called directly as a function."""
        self.log(message)


def create_logger(log_file: Optional[str] = None, prefix: str = "") -> AgentLogger:
    """
    Factory function to create a logger.
    
    Args:
        log_file: Optional path to log file
        prefix: Optional prefix for all messages
    
    Returns:
        Configured AgentLogger instance
    """
    return AgentLogger(log_file=log_file, prefix=prefix)


# Convenience function for simple logging without a logger instance
def log_to_file(message: str, log_file: Optional[str] = None) -> None:
    """
    Simple one-shot logging function.
    
    Args:
        message: Message to log
        log_file: Optional file to also write to
    """
    print(message)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")
