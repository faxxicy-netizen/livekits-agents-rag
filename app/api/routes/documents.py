import asyncio

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.api.context import AppContext
from app.api.dependencies import get_context, owner
from app.knowledge.documents import MAX_UPLOAD_BYTES, DocumentError, build_document, extract_text
from app.knowledge.vector import VectorError
from app.models import DocumentInput, DocumentUpdate

router = APIRouter()


@router.get("/api/documents")
def documents(knowledge_base_id: str = "default", ctx: AppContext = Depends(get_context)):
    ctx.knowledge(knowledge_base_id)
    return ctx.registry.repository.list(knowledge_base_id=knowledge_base_id)


@router.get("/api/documents/{document_id}")
def document(document_id: str, ctx: AppContext = Depends(get_context)):
    try:
        doc = ctx.registry.repository.get(document_id)
    except KeyError:
        raise HTTPException(404, "文档不存在。")
    return {**ctx.registry.repository.summary(doc), "content": doc["content"]}


async def save_document(
    filename, content, old=None, knowledge_base_id="default", *, ctx: AppContext
):
    try:
        ctx.knowledge(knowledge_base_id)
        doc = build_document(filename, content, old["id"] if old else None, knowledge_base_id)
        saved = await ctx.registry.manager.save(doc, old)
        ctx.registry.reload()
        return {
            "status": "indexed",
            "document": saved,
            "documents": len(ctx.registry.repository.list()),
        }
    except (DocumentError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from None
    except VectorError as exc:
        raise HTTPException(503, str(exc)) from None


@router.post("/api/documents")
async def upload_document(
    body: DocumentInput, identity=Depends(owner), ctx: AppContext = Depends(get_context)
):
    ctx.knowledge(body.knowledge_base_id)
    async with ctx.registry.manager.lock:
        if ctx.registry.repository.by_filename(body.filename, body.knowledge_base_id):
            raise HTTPException(409, "已有同名文档，请打开原文档后编辑，避免覆盖。")
        return await save_document(
            body.filename, body.content, knowledge_base_id=body.knowledge_base_id, ctx=ctx
        )


@router.post("/api/documents/upload")
async def upload_file(
    file: UploadFile = File(...),
    knowledge_base_id: str = Form("default"),
    identity=Depends(owner),
    ctx: AppContext = Depends(get_context),
):
    ctx.knowledge(knowledge_base_id)
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    try:
        content = await asyncio.to_thread(extract_text, file.filename or "", data)
    except DocumentError as exc:
        raise HTTPException(400, str(exc)) from None
    finally:
        await file.close()
    async with ctx.registry.manager.lock:
        if ctx.registry.repository.by_filename(file.filename, knowledge_base_id):
            raise HTTPException(409, "已有同名文档，请打开原文档后编辑，避免覆盖。")
        return await save_document(
            file.filename, content, knowledge_base_id=knowledge_base_id, ctx=ctx
        )


@router.put("/api/documents/{document_id}")
async def update_document(
    document_id: str,
    body: DocumentUpdate,
    identity=Depends(owner),
    ctx: AppContext = Depends(get_context),
):
    async with ctx.registry.manager.lock:
        try:
            old = ctx.registry.repository.get(document_id)
        except KeyError:
            raise HTTPException(404, "文档不存在。") from None
        if body.expected_version and body.expected_version != old["version"]:
            raise HTTPException(409, "文档已被更新，请重新打开后再编辑。")
        return await save_document(
            old["filename"], body.content, old, old["knowledge_base_id"], ctx=ctx
        )


@router.delete("/api/documents/{document_id}")
async def delete_document(
    document_id: str, identity=Depends(owner), ctx: AppContext = Depends(get_context)
):
    async with ctx.registry.manager.lock:
        try:
            await ctx.registry.manager.delete(document_id)
        except KeyError:
            raise HTTPException(404, "文档不存在。") from None
        except VectorError as exc:
            raise HTTPException(503, str(exc)) from None
        ctx.registry.reload()
        return {"status": "deleted", "document_id": document_id}
