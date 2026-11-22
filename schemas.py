"""Pydantic-схеми структурованих відповідей агента."""

from pydantic import BaseModel, Field


class FinalAnswer(BaseModel):
    """Структурована фінальна відповідь DevOps-агента."""

    answer: str = Field(
        min_length=1,
        description="Підсумкова відповідь українською мовою"
    )

    confidence: float = Field(
        ge=0,
        le=1,
        description="Оцінка впевненості моделі від 0 до 1"
    )

    sources: list[str] = Field(
        description="Назви інструментів, на результатах яких базується відповідь"
    )