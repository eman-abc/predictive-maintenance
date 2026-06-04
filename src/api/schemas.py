"""API request/response models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    ollama_available: bool = False
    ollama_model: str | None = None
    datasets_available: list[str] = Field(default_factory=list)


class BriefingRequest(BaseModel):
    mode: str = "instant"
    dataset_id: str = "FD001"
    context: dict[str, Any] = Field(default_factory=dict)


class BriefingResponse(BaseModel):
    text: str
    source: str
    mode: str
    error: str | None = None


class MetricExplainRequest(BaseModel):
    section_id: str
    dataset_id: str = "FD001"
    summary: dict[str, Any] = Field(default_factory=dict)
    registry_entry: dict[str, Any] | None = None


class MetricExplainResponse(BaseModel):
    text: str
    source: str
    section_id: str
    error: str | None = None


class WorkOrdersRequest(BaseModel):
    dataset_id: str = "FD001"
    levels: list[str] = Field(default_factory=lambda: ["critical", "warning"])


class WorkOrdersResponse(BaseModel):
    results: list[dict[str, Any]]
    count: int
