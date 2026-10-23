from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Question(StrictModel):
    question: str = Field(min_length=3, max_length=2000)

    @field_validator("question")
    @classmethod
    def not_blank(cls, value):
        if len(value.strip()) < 3:
            raise ValueError("Veuillez saisir une question.")
        return value.strip()


class Identity(StrictModel):
    user_id: str
    roles: list[str]


class Claim(StrictModel):
    text: str = Field(min_length=1, max_length=600)
    source_id: int = Field(ge=1, le=8)
    quote: str = Field(min_length=10, max_length=800)


class Draft(StrictModel):
    supported: bool
    claims: list[Claim] = Field(max_length=3)

    @model_validator(mode="after")
    def coherent(self):
        if self.supported != bool(self.claims):
            raise ValueError("Une réponse étayée nécessite au moins une affirmation sourcée.")
        return self


class GenerationOutput(StrictModel):
    evidence_ids: list[int] = Field(max_length=3)


class Citation(StrictModel):
    label: str
    document_id: str
    document: str
    version: str
    page: int
    quote: str
    score: float


class Answer(StrictModel):
    status: Literal["answered", "abstained"]
    answer: str
    citations: list[Citation]
    reason: str | None = None
    notice: str = "Réponse documentaire à vérifier par un technicien habilité. Aucun équipement n'est piloté."


class ImageQuestion(Question):
    image_base64: str = Field(min_length=4, max_length=7 * 1024 * 1024)
    equipment_reference: str | None = Field(default=None, min_length=2, max_length=120)


class ImageAnswer(StrictModel):
    status: Literal["answered", "abstained", "clarification_required"]
    observation: str
    observation_language: str = "en"
    observation_verified: bool = False
    model: str
    image_sha256: str
    equipment_reference: str | None
    clarification: str | None
    uncertainties: list[str]
    documentary_answer: Answer


class MultimodalQuestion(Question):
    image_base64: str | None = Field(default=None, min_length=4, max_length=7 * 1024 * 1024)
    equipment_reference: str | None = Field(default=None, min_length=2, max_length=120)
    top_k: int = Field(default=5, ge=1, le=8)


class FigureMatch(StrictModel):
    figure_id: str
    kind: str
    bbox: list[float]
    scores: dict[str, float] = Field(default_factory=dict)


class MatchedPage(StrictModel):
    document_id: str
    document: str
    version: str
    page: int
    fusion_score: float
    modalities: list[str]
    text_excerpts: list[str]
    figures: list[FigureMatch]


class MultimodalResult(StrictModel):
    pages: list[MatchedPage]
    figures_index_status: Literal["ready", "empty", "not_indexed"]
    notice: str = "Correspondances documentaires à vérifier. La ressemblance visuelle ne prouve pas la signification d'un symbole ou d'un voyant."
