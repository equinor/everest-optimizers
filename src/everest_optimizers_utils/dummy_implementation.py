#!/usr/bin/env python3
# src/everest_optimizers_utils/dummy_implementation.py

from typing import Any, Dict, List, Optional, Union

# ropt.config
class EnOptConfig:
    def __init__(self, *args, **kwargs):
        pass

# ropt.enums
class EventType:
    pass

class ExitCode:
    pass

# ropt.plan
class BasicOptimizer:
    def __init__(self, *args, **kwargs):
        pass

class Event:
    def __init__(self, *args, **kwargs):
        pass

# ropt.plugins
class PluginManager:
    def __init__(self, *args, **kwargs):
        pass

# ropt.results
class FunctionResults:
    def __init__(self, *args, **kwargs):
        pass

class GradientResults:
    def __init__(self, *args, **kwargs):
        pass

class Results:
    def __init__(self, *args, **kwargs):
        pass

# ropt.transforms
class OptModelTransforms:
    def __init__(self, *args, **kwargs):
        pass

# ropt.transforms.base
class NonLinearConstraintTransform:
    def __init__(self, *args, **kwargs):
        pass

class ObjectiveTransform:
    def __init__(self, *args, **kwargs):
        pass

# ropt_dakota.dakota
_SUPPORTED_METHODS = {"conmin_mfd", "conmin_frcg", "soga", "optpp_q_newton", "moga", "coliny_ea"}  # Dummy implementation of the variable as a set
