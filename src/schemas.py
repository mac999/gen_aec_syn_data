"""
Pydantic schemas for sLLM (SFT) and VLM dataset samples.
These mirror the JSONL schemas defined in the PRD §3.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class EvidenceBlock(BaseModel):
    doc_id: str = Field(description="Source document identifier")
    section: str = Field(description="Section / clause reference within the document")
    answer_span: Optional[str] = Field(default=None, description="Verbatim text span that grounds the answer")


class SFTInputMetadata(BaseModel):
    project_type: str = Field(description="e.g. 교량, 건물, 터널")
    language: str = Field(default="ko", description="Language code, e.g. ko")


class SFTInput(BaseModel):
    context: str = Field(description="Relevant document chunk or clause text")
    metadata: SFTInputMetadata


class SFTOutput(BaseModel):
    answer: str = Field(description="Answer to the instruction")
    evidence: List[EvidenceBlock] = Field(
        description="Grounding evidence blocks linking answer to source"
    )
    final_label: str = Field(
        default="answerable",
        description="compliant | non_compliant | answerable | unanswerable | ambiguous",
    )


class SFTSample(BaseModel):
    id: str = Field(description="Unique sample ID, e.g. sft_000001")
    task_type: str = Field(
        default="regulation_qa",
        description="Task category: regulation_qa | numeric_judgment | risk_description",
    )
    domain_tags: List[str] = Field(
        default_factory=list,
        description="Domain tags, e.g. ['구조', '안전']",
    )
    source_doc_ids: List[str] = Field(
        default_factory=list,
        description="Source document IDs referenced in this sample",
    )
    instruction: str = Field(description="Natural language question / instruction")
    input: SFTInput
    output: SFTOutput

    def to_jsonl_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")

class VLMMetadata(BaseModel):
    project_type: str = Field(description="e.g. 교량, 건물, 터널")
    bim_element_ids: List[str] = Field(description="IFC global IDs or element GUIDs")
    trade_type: str = Field(description="e.g. 철근콘크리트, 철골, 목구조")
    view_type: str = Field(description="3d_render | top_view | front_view | perspective")


class VLMOutput(BaseModel):
    answer: str = Field(description="Free-text answer to the instruction")
    label: str = Field(
        description="match | partial_match | mismatch | unknown"
    )
    evidence: List[str] = Field(
        description="Bullet-point evidence sentences supporting the answer"
    )


class VLMSample(BaseModel):
    id: str = Field(description="Unique sample ID, e.g. vlm_000001")
    task_type: str = Field(
        default="bim_site_alignment",
        description="Task category: bim_site_alignment | element_detection | progress_assessment",
    )
    images: List[str] = Field(
        description="Relative paths under ./output/images/ — [bim_render, site_photo]"
    )
    metadata: VLMMetadata
    instruction: str = Field(description="Natural language task instruction for the VLM")
    output: VLMOutput

    def to_jsonl_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")

class DocumentChunk(BaseModel):
    doc_id: str
    chunk_index: int
    page_numbers: List[int]
    text: str
    char_count: int

class DAPTSample(BaseModel):
    id: str = Field(description="Row 고유 ID")
    doc_id: str = Field(description="원본 문서 ID")
    source_type: str = Field(default="", description="법령/시방서/회의록/보고서 등")
    source_name: str = Field(default="", description="문서명")
    source_org: str = Field(default="", description="발행기관/작성기관")
    source_date: str = Field(default="", description="작성일 또는 개정일")
    language: str = Field(default="ko", description="ko / en / mixed")
    domain_tags: List[str] = Field(default_factory=list, description="구조/토목/건축/안전/BIM 등")
    project_type: str = Field(default="", description="도로/교량/건축/플랜트 등")
    text: str = Field(description="학습 본문")
    section_path: str = Field(default="", description="장/절/조항 경로")
    page_range: str = Field(default="", description="원본 페이지 범위")
    license: str = Field(default="", description="활용 가능 라이선스 정보")
    raw_hash: str = Field(default="", description="중복 체크용 hash")

    def to_jsonl_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")

class IFCElementInfo(BaseModel):
    global_id: str
    ifc_type: str
    name: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)
    render_path: Optional[str] = None  # relative path to the rendered image
