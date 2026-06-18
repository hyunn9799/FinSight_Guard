"""Repository for source documents (with revision lineage) and chunks."""

from datetime import datetime

from src.db.models import DocumentChunk, SourceDocument
from src.db.repositories.base import BaseRepository


class SourceDocumentRepository(BaseRepository):
    def add_document(
        self,
        *,
        document_type: str,
        source_name: str,
        content_hash: str,
        collected_at: datetime,
        ticker_id=None,
        source_url: str | None = None,
        title: str | None = None,
        language: str | None = None,
        published_at: datetime | None = None,
        raw_content_ref: str | None = None,
        metadata: dict | None = None,
        revision_group_id=None,
        supersedes_document_id=None,
        status: str = "active",
    ) -> SourceDocument:
        kwargs = dict(
            document_type=document_type,
            source_name=source_name,
            content_hash=content_hash,
            collected_at=collected_at,
            ticker_id=ticker_id,
            source_url=source_url,
            title=title,
            language=language,
            published_at=published_at,
            raw_content_ref=raw_content_ref,
            doc_metadata=metadata or {},
            supersedes_document_id=supersedes_document_id,
            status=status,
        )
        if revision_group_id is not None:
            kwargs["revision_group_id"] = revision_group_id
        document = SourceDocument(**kwargs)
        self.session.add(document)
        self.session.flush()
        return document

    def add_correction(
        self, prior: SourceDocument, *, content_hash: str, collected_at: datetime, **fields
    ) -> SourceDocument:
        return self.add_document(
            document_type=fields.pop("document_type", prior.document_type),
            source_name=fields.pop("source_name", prior.source_name),
            content_hash=content_hash,
            collected_at=collected_at,
            ticker_id=fields.pop("ticker_id", prior.ticker_id),
            revision_group_id=prior.revision_group_id,
            supersedes_document_id=prior.id,
            **fields,
        )

    def add_chunk(
        self,
        source_document_id,
        chunk_index: int,
        chunk_text: str,
        chunk_hash: str,
        *,
        token_count: int | None = None,
        metadata: dict | None = None,
    ) -> DocumentChunk:
        chunk = DocumentChunk(
            source_document_id=source_document_id,
            chunk_index=chunk_index,
            chunk_text=chunk_text,
            chunk_hash=chunk_hash,
            token_count=token_count,
            chunk_metadata=metadata or {},
        )
        self.session.add(chunk)
        self.session.flush()
        return chunk

    def list_chunks(self, source_document_id) -> list[DocumentChunk]:
        return (
            self.session.query(DocumentChunk)
            .filter(DocumentChunk.source_document_id == source_document_id)
            .order_by(DocumentChunk.chunk_index)
            .all()
        )
