import uvicorn
from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from llm_api.settings import fastapi_options


app = FastAPI(docs_url=None, **fastapi_options)

@app.get("/api/v1/heartbeat")
async def is_alive():
    return {}


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_htlm():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title,
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,

    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)