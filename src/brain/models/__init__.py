"""Núcleos neurais. Pesos sempre iniciados aleatoriamente; nada pré-treinado."""

from brain.models.registry import build_model, count_parameters

__all__ = ["build_model", "count_parameters"]
