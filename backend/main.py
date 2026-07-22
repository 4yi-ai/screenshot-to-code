# Load environment variables first
from dotenv import load_dotenv

load_dotenv()


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import IS_DEBUG_ENABLED
from routes import (
    screenshot,
    generate_code,
    home,
    export,
    design_systems,
    prompt_reports,
)
from uploaded_assets import configure_uploaded_asset_routes

app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)
configure_uploaded_asset_routes(app)


@app.on_event("startup")
async def log_debug_mode() -> None:
    debug_status = "ENABLED" if IS_DEBUG_ENABLED else "DISABLED"
    print(f"Backend startup complete. Debug mode is {debug_status}.")


@app.on_event("startup")
async def probe_screenshot_preview_on_startup() -> None:
    # Detect (and warm up) headless Chromium so the screenshot_preview tool is
    # only offered when it can actually run. Logs the outcome.
    from preview_screenshot import probe_screenshot_preview

    await probe_screenshot_preview()

# Configure CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add routes
app.include_router(generate_code.router)
app.include_router(screenshot.router)
app.include_router(home.router)
app.include_router(export.router)
app.include_router(design_systems.router)
app.include_router(prompt_reports.router)

# --- 4yi single-container serving ---
import os
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles


@app.get("/healthz")
async def healthz() -> JSONResponse:
    return JSONResponse({"ok": True})


_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_STATIC_DIR):
    # Registered last so all API/WS/healthz routes win over the SPA catch-all.
    app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="spa")
