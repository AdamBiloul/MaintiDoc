from contextlib import asynccontextmanager
import logging
from starlette.concurrency import run_in_threadpool

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from maintidoc.auth import authenticate
from maintidoc.rag import RagEngine
from maintidoc.schemas import Answer, Question, ImageQuestion, ImageAnswer, MultimodalQuestion, MultimodalResult
from maintidoc.settings import Settings
from maintidoc.vision import PhotoAssistant, InvalidImage, MAX_BODY_BYTES
from maintidoc.multimodal import MultimodalExplorer, StaleFigureIndex

logger = logging.getLogger("maintidoc")


def create_app(settings=None, engine=None, photo=None, explorer=None):
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app):
        app.state.engine = engine or RagEngine(settings)
        app.state.photo = photo or PhotoAssistant(app.state.engine, settings)
        app.state.explorer = explorer or MultimodalExplorer(app.state.engine, settings)
        if isinstance(app.state.engine, RagEngine):
            app.state.engine.figure_store = app.state.explorer.figures
        try:
            yield
        finally:
            app.state.explorer.close()
            app.state.engine.close()

    app = FastAPI(title="MaintiDoc — V2 prototype", version="2.0.0", lifespan=lifespan)
    bearer = HTTPBearer(auto_error=False)

    def identity(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
        try:
            user = authenticate(credentials.credentials if credentials else None, settings.root)
        except (OSError, ValueError, KeyError):
            raise HTTPException(503, "Identités non configurées.")
        if user is None:
            raise HTTPException(401, "Authentification requise.", headers={"WWW-Authenticate": "Bearer"})
        return user

    @app.get("/health")
    def health():
        return {"status": "ok", "version": "2.0.0"}

    async def bounded_json(request, schema):
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > MAX_BODY_BYTES:
                raise HTTPException(413, "Requête supérieure à 8 Mo.")
            body.extend(chunk)
        try:
            return schema.model_validate_json(bytes(body))
        except ValueError:
            raise HTTPException(422, "Question ou image encodée invalide.")

    @app.get("/documents")
    def documents(user=Depends(identity)):
        return [{k: d[k] for k in ("id", "title", "version", "pages", "family")}
                for d in app.state.engine.documents(user)]

    @app.get("/documents/{document_id}/pdf")
    def document_pdf(document_id: str, user=Depends(identity)):
        try:
            path = app.state.engine.pdf_path(document_id, user)
        except FileNotFoundError:
            raise HTTPException(404, "Document inaccessible.")
        except ValueError:
            raise HTTPException(409, "Document modifié : réindexation nécessaire.")
        return FileResponse(path, media_type="application/pdf", filename=path.name)

    @app.post("/ask", response_model=Answer)
    def ask(request: Question, user=Depends(identity)):
        try:
            return app.state.engine.ask(request.question, user)
        except httpx.HTTPError:
            raise HTTPException(503, "Modèle génératif indisponible. Vérifiez Ollama et le modèle local.")
        except (RuntimeError, OSError, TimeoutError):
            raise HTTPException(503, "Recherche documentaire indisponible. Vérifiez les modèles locaux.")
        except Exception:
            # Ne journalise ni questions, ni passages, ni jetons.
            logger.error("Echec interne du workflow documentaire.")
            raise HTTPException(500, "Erreur du workflow documentaire.")

    @app.post("/analyze-image", response_model=ImageAnswer, openapi_extra={
        "requestBody": {"required": True, "content": {"application/json": {"schema": ImageQuestion.model_json_schema()}}}
    })
    async def analyze_image(request: Request, user=Depends(identity)):
        question = await bounded_json(request, ImageQuestion)
        try:
            return await run_in_threadpool(app.state.photo.analyze, question, user)
        except InvalidImage as error:
            raise HTTPException(400, str(error))
        except StaleFigureIndex as error:
            raise HTTPException(409, str(error))
        except (RuntimeError, OSError, TimeoutError):
            raise HTTPException(503, "Analyse visuelle indisponible. Vérifiez le modèle local et la mémoire disponible.")
        except httpx.HTTPError:
            raise HTTPException(503, "Génération documentaire indisponible.")

    @app.post("/search-multimodal", response_model=MultimodalResult, openapi_extra={
        "requestBody": {"required": True, "content": {"application/json": {"schema": MultimodalQuestion.model_json_schema()}}}
    })
    async def search_multimodal(request: Request, user=Depends(identity)):
        question = await bounded_json(request, MultimodalQuestion)
        try:
            return await run_in_threadpool(app.state.explorer.search, question, user)
        except InvalidImage as error:
            raise HTTPException(400, str(error))
        except StaleFigureIndex as error:
            raise HTTPException(409, str(error))
        except (RuntimeError, OSError, TimeoutError):
            raise HTTPException(503, "Recherche des figures indisponible. Vérifiez les modèles et l'index local.")

    @app.get("/figures/{figure_id}/image")
    def figure_image(figure_id: str, user=Depends(identity)):
        try:
            path = app.state.explorer.figures.asset_path(figure_id, user)
        except FileNotFoundError:
            raise HTTPException(404, "Figure inaccessible.")
        except (StaleFigureIndex, ValueError):
            raise HTTPException(409, "Index des figures modifié : réindexation nécessaire.")
        return FileResponse(path, media_type="image/png", filename=f"{figure_id}.png")

    return app


app = create_app()
