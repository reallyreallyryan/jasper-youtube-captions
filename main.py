from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI
app = FastAPI(title="Jasper Caption Generator", description="AI-powered caption generation for healthcare marketing")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse("static/index.html")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "message": "Minimal version deployed successfully!",
        "video_processor": False,
        "youtube_processor": False
    }

@app.post("/process-videos")
async def process_videos():
    return {"error": "Video processing not available in minimal version"}

@app.post("/process-urls") 
async def process_urls():
    return {"error": "YouTube processing not available in minimal version"}

@app.post("/process-csv")
async def process_csv():
    return {"error": "CSV processing not available in minimal version"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)