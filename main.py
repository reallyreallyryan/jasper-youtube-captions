from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
import csv
import json
import asyncio
from pathlib import Path
from typing import List, Optional
import subprocess
from openai import OpenAI
import time
from datetime import datetime

class YouTubeCaptionGenerator:
    def __init__(self, openai_api_key=None):
        self.client = OpenAI(api_key=openai_api_key or os.getenv('OPENAI_API_KEY'))
        
        # Check if yt-dlp is available
        try:
            subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
            self.yt_dlp_available = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.yt_dlp_available = False
            print("⚠️ yt-dlp not available - some features may be limited")
    
    def process_shorts_url(self, youtube_url):
        try:
            if 'youtube.com/shorts/' not in youtube_url and 'youtu.be/' not in youtube_url:
                return {'success': False, 'error': 'Invalid YouTube URL'}
            
            if not self.yt_dlp_available:
                return {'success': False, 'error': 'YouTube processing not available on this platform'}
            
            transcript = self._extract_transcript(youtube_url)
            if not transcript:
                return {'success': False, 'error': 'Could not extract transcript'}
            
            caption = self._generate_caption(transcript)
            
            return {
                'success': True,
                'caption': caption,
                'transcript': transcript[:200] + '...' if len(transcript) > 200 else transcript
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _extract_transcript(self, youtube_url):
        # Try auto-transcript first
        transcript = self._get_auto_transcript(youtube_url)
        if transcript:
            return transcript
        
        # For now, skip audio download on Railway to avoid complexity
        return None
    
    def _get_auto_transcript(self, youtube_url):
        try:
            cmd = [
                'yt-dlp', '--write-auto-subs', '--write-subs', '--skip-download',
                '--sub-format', 'vtt', '--sub-langs', 'en', 
                youtube_url
            ]
            
            with tempfile.TemporaryDirectory() as temp_dir:
                result = subprocess.run(cmd, cwd=temp_dir, capture_output=True, text=True, timeout=30)
                vtt_files = list(Path(temp_dir).glob('*.vtt'))
                
                if vtt_files:
                    with open(vtt_files[0], 'r', encoding='utf-8') as f:
                        vtt_content = f.read()
                    
                    lines = vtt_content.split('\n')
                    transcript_lines = []
                    
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('WEBVTT') and '-->' not in line and not line.isdigit():
                            transcript_lines.append(line)
                    
                    return ' '.join(transcript_lines)
        except Exception as e:
            print(f"Transcript extraction failed: {e}")
        return None
    
    def _generate_caption(self, transcript):
        if not transcript:
            return "❌ No transcript available"
        
        prompt = f"""
You are a social media expert specializing in healthcare marketing. 

Create a catchy, engaging caption for a YouTube Short based on this transcript.

TRANSCRIPT:
{transcript}

REQUIREMENTS:
- 1-2 punchy sentences maximum
- Healthcare/medical tone but accessible to general audience  
- Include relevant emojis (2-3 max)
- Focus on the key insight or takeaway
- Make it shareable and engaging
- Avoid medical jargon - keep it conversational

EXAMPLES OF GOOD STYLE:
"🩺 Did you know this simple trick can reduce back pain in 30 seconds? Your spine will thank you!"
"💊 The truth about supplements that Big Pharma doesn't want you to know..."
"🏥 This doctor's 5-minute morning routine prevents 90% of common illnesses"

Generate ONLY the caption, no explanation:
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are an expert social media caption writer for healthcare content."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.8
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"❌ Caption error: {str(e)}"

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

# Initialize processors
youtube_generator = None

@app.on_event("startup")
async def startup_event():
    global youtube_generator
    try:
        youtube_generator = YouTubeCaptionGenerator()
        print("✅ YouTube caption generator initialized!")
    except Exception as e:
        print(f"⚠️ Warning: YouTube generator failed to initialize: {e}")

@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse("static/index.html")

@app.post("/process-videos")
async def process_videos(files: List[UploadFile] = File(...)):
    """Process uploaded video files - TEMPORARILY DISABLED"""
    return {
        "results": [
            {
                "filename": file.filename,
                "success": False,
                "caption": "🚧 Video processing coming soon! Use YouTube URLs for now.",
                "transcript_preview": "",
                "error": "Video processing temporarily unavailable on Railway. Use YouTube Shorts URLs instead!",
                "processing_time": 0
            }
            for file in files
        ]
    }

@app.post("/process-urls")
async def process_urls(urls: List[str]):
    """Process YouTube Shorts URLs"""
    if not youtube_generator:
        raise HTTPException(status_code=500, detail="YouTube processor not available")
    
    results = []
    
    for url in urls:
        start_time = time.time()
        
        try:
            result = youtube_generator.process_shorts_url(url.strip())
            processing_time = time.time() - start_time
            
            results.append({
                "url": url,
                "success": result['success'],
                "caption": result.get('caption', ''),
                "transcript_preview": result.get('transcript', ''),
                "error": result.get('error'),
                "processing_time": round(processing_time, 2)
            })
            
        except Exception as e:
            processing_time = time.time() - start_time
            results.append({
                "url": url,
                "success": False,
                "caption": '',
                "transcript_preview": '',
                "error": str(e),
                "processing_time": round(processing_time, 2)
            })
    
    return {"results": results}

@app.post("/process-csv")
async def process_csv(file: UploadFile = File(...)):
    """Process CSV file with URLs"""
    if not youtube_generator:
        raise HTTPException(status_code=500, detail="YouTube processor not available")
    
    # Save uploaded CSV
    with tempfile.NamedTemporaryFile(mode='w+b', delete=False, suffix='.csv') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    try:
        # Read CSV
        with open(tmp_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            rows = list(reader)
        
        # Find URL column
        url_columns = [col for col in headers if 'url' in col.lower()]
        if not url_columns:
            raise HTTPException(status_code=400, detail="No URL column found in CSV")
        
        url_column = url_columns[0]
        
        # Process each row
        enhanced_rows = []
        for row in rows:
            url = row.get(url_column, '').strip()
            
            if url and 'youtube.com/shorts/' in url:
                result = youtube_generator.process_shorts_url(url)
                
                enhanced_row = row.copy()
                enhanced_row.update({
                    'ai_caption': result.get('caption', '❌ Processing failed'),
                    'ai_transcript_preview': result.get('transcript', ''),
                    'ai_status': 'success' if result['success'] else 'failed'
                })
            else:
                enhanced_row = row.copy()
                enhanced_row.update({
                    'ai_caption': '❌ Invalid YouTube Shorts URL',
                    'ai_transcript_preview': '',
                    'ai_status': 'invalid_url'
                })
            
            enhanced_rows.append(enhanced_row)
        
        # Create output CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"enhanced_captions_{timestamp}.csv"
        output_path = f"/tmp/{output_filename}"
        
        new_headers = list(headers) + ['ai_caption', 'ai_transcript_preview', 'ai_status']
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=new_headers)
            writer.writeheader()
            writer.writerows(enhanced_rows)
        
        return FileResponse(
            output_path,
            media_type='text/csv',
            filename=output_filename
        )
        
    finally:
        # Cleanup temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "video_processor": False,  # Disabled for now
        "youtube_processor": youtube_generator is not None,
        "timestamp": datetime.now().isoformat(),
        "platform": "Railway",
        "features": ["YouTube URL processing", "CSV enhancement"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)