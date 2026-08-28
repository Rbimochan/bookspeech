from pydantic import BaseModel


class Chapter(BaseModel):
    index: int
    title: str
    text: str


class Book(BaseModel):
    title: str
    author: str
    cover_path: str | None
    language: str | None
    chapters: list[Chapter]


class Chunk(BaseModel):
    chapter_index: int
    chunk_index: int
    text: str
    paragraph_start: int
    paragraph_end: int
