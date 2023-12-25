import uvicorn

from fastapi import APIRouter, FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger

from app.config import settings
from app.api import api_router
from app import  __version__

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_STR}/openapi.json"
)

#app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

root_router = APIRouter()

@root_router.get("/")
def index(request: Request):
    return templates.TemplateResponse("home.html", 
                                      {"request": request,
                                       "api_version": __version__})

app.include_router(api_router, prefix=settings.API_STR)
app.include_router(root_router)

if __name__ == "__main__":
    # Use this for debugging purposes only
    logger.warning("Running in development mode. Do not run like this in production.")

    uvicorn.run(app, host='localhost', port=8002, log_level='debug')


