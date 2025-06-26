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
import shutil

# Import our processors (full-featured versions)
from dataclasses import dataclass

@dataclass
class ProcessingResult:
    filename: str
    success: bool
    caption: str
    transcript_preview: str
    error: str = None
    processing_time: float = 0

class LocalVideoCaptionGenerator:
    def __init__(self, openai_api_key=None):
        self.client = OpenAI(api_key=openai_api_key or os.getenv('OPENAI_API_KEY'))
        
        # Check if ffmpeg is installed
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            self.ffmpeg_available = True
            print("✅ ffmpeg available for video processing")
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.ffmpeg_available = False
            print("⚠️ ffmpeg not found - video processing disabled")
    
    def process_video_file(self, video_path):
        try:
            if not self.ffmpeg_available:
                return {'success': False, 'error': 'ffmpeg not available on this platform'}
            
            print(f"🎬 Processing video: {video_path}")
            
            # Extract audio
            audio_path = self._extract_audio(video_path)
            if not audio_path:
                return {'success': False, 'error': 'Could not extract audio'}
            
            # Transcribe
            transcript = self._transcribe_audio(audio_path)
            if os.path.exists(audio_path):
                os.unlink(audio_path)
            
            if not transcript:
                return {'success': False, 'error': 'Could not transcribe audio'}
            
            # Generate caption
            caption = self._generate_caption(transcript)
            
            return {
                'success': True,
                'caption': caption,
                'transcript': transcript[:200] + '...' if len(transcript) > 200 else transcript
            }
            
        except Exception as e:
            print(f"❌ Video processing error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _extract_audio(self, video_path):
        try:
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                audio_path = temp_file.name
            
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vn', '-acodec', 'mp3', '-ar', '16000', '-ac', '1', '-y',
                audio_path
            ]
            
            print("🎵 Extracting audio...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0 and os.path.exists(audio_path):
                print("✅ Audio extracted!")
                return audio_path
            else:
                print(f"❌ Audio extraction failed: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"⚠️ Audio extraction error: {str(e)}")
            return None
    
    def _transcribe_audio(self, audio_path):
        try:
            print("🎤 Transcribing with Whisper...")
            with open(audio_path, 'rb') as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            print("✅ Transcription complete!")
            return transcript
        except Exception as e:
            print(f"⚠️ Transcription failed: {str(e)}")
            return None
    
    def _generate_caption(self, transcript):
        if not transcript:
            return "❌ No transcript available"
        
        prompt = f"""
You are a professional healthcare content strategist creating social media captions.

Create an engaging but professional caption for a healthcare video based on this transcript.

TRANSCRIPT:
{transcript}

REQUIREMENTS:
- 1-2 sentences maximum
- Professional healthcare tone but accessible language
- Include 1-2 relevant, professional emojis (avoid excessive emoji use)
- Focus on the clinical benefit or key insight
- Maintain credibility while being engaging
- Avoid overly casual language or "hack" terminology
- Use evidence-based language when possible

EXAMPLES OF GOOD PROFESSIONAL TONE:
"🩺 New research shows this technique reduces chronic back pain by 40% in clinical trials."
"💡 Understanding bone marrow concentrate therapy: a proven approach to accelerating healing."
"🔬 Here's what the latest studies reveal about non-surgical pain management options."

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
            return response.choices[0].message.content.strip().replace('"', '')
        except Exception as e:
            return f"❌ Caption error: {str(e)}"

class YouTubeCaptionGenerator:
    def __init__(self, openai_api_key=None):
        self.client = OpenAI(api_key=openai_api_key or os.getenv('OPENAI_API_KEY'))
        
        try:
            subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
            self.yt_dlp_available = True
            print("✅ yt-dlp available for YouTube processing")
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.yt_dlp_available = False
            print("⚠️ yt-dlp not available")
    
    def process_shorts_url(self, youtube_url):
        try:
            if 'youtube.com/shorts/' not in youtube_url and 'youtu.be/' not in youtube_url:
                return {'success': False, 'error': 'Invalid YouTube URL'}
            
            if not self.yt_dlp_available:
                return {'success': False, 'error': 'YouTube processing not available'}
            
            print(f"🎥 Processing YouTube: {youtube_url}")
            transcript = self._extract_transcript(youtube_url)
            if not transcript:
                return {
                    'success': False, 
                    'error': 'No transcript or captions available. Try a different video with auto-generated captions.'
                }
            
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
            print("✅ Found auto-generated transcript!")
            return transcript
        
        # Fallback: Download audio and transcribe with Whisper
        print("📥 No auto-captions, downloading audio for transcription...")
        audio_path = self._download_audio(youtube_url)
        if audio_path:
            transcript = self._transcribe_audio(audio_path)
            if os.path.exists(audio_path):
                os.unlink(audio_path)
            return transcript
        
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
            print(f"⚠️ Auto-transcript failed: {str(e)}")
        return None
    
    def _download_audio(self, youtube_url):
        try:
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_path = temp_file.name
            
            cmd = [
                'yt-dlp', '-x', '--audio-format', 'mp3',
                '--audio-quality', '0', '-o', temp_path.replace('.mp3', '.%(ext)s'),
                youtube_url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                possible_files = [temp_path, temp_path.replace('.mp3', '.m4a'), temp_path.replace('.mp3', '.webm')]
                for file_path in possible_files:
                    if os.path.exists(file_path):
                        return file_path
        except Exception as e:
            print(f"⚠️ Audio download failed: {str(e)}")
        return None
    
    def _transcribe_audio(self, audio_path):
        try:
            print("🎤 Transcribing with Whisper...")
            with open(audio_path, 'rb') as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            return transcript
        except Exception as e:
            print(f"⚠️ Transcription failed: {str(e)}")
            return None
    
    def _generate_caption(self, transcript):
        if not transcript:
            return "❌ No transcript available"
        
        prompt = f"""
You are a professional healthcare content strategist creating social media captions.

Create an engaging but professional caption for a healthcare video based on this transcript.

TRANSCRIPT:
{transcript}

REQUIREMENTS:
- 1-2 sentences maximum
- Professional healthcare tone but accessible language
- Include 1-2 relevant, professional emojis (avoid excessive emoji use)
- Focus on the clinical benefit or key insight
- Maintain credibility while being engaging
- Avoid overly casual language or "hack" terminology
- Use evidence-based language when possible

EXAMPLES OF GOOD PROFESSIONAL TONE:
"🩺 New research shows this technique reduces chronic back pain by 40% in clinical trials."
"💡 Understanding bone marrow concentrate therapy: a proven approach to accelerating healing."
"🔬 Here's what the latest studies reveal about non-surgical pain management options."

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
            return response.choices[0].message.content.strip().replace('"', '')
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
video_generator = None
youtube_generator = None

@app.on_event("startup")
async def startup_event():
    global video_generator, youtube_generator
    print("🚀 Starting up Jasper Caption Generator...")
    
    try:
        print("📹 Initializing video processor...")
        video_generator = LocalVideoCaptionGenerator()
        print("✅ Video processor ready!")
    except Exception as e:
        print(f"⚠️ Video processor failed: {e}")
        video_generator = None
    
    try:
        print("🎬 Initializing YouTube processor...")
        youtube_generator = YouTubeCaptionGenerator()
        print("✅ YouTube processor ready!")
    except Exception as e:
        print(f"⚠️ YouTube processor failed: {e}")
        youtube_generator = None
    
    print("🎉 Startup complete!")

@app.get("/")
async def root():
    try:
        return FileResponse("static/index.html")
    except:
        return {"message": "Jasper Caption Generator is running!", "status": "ok"}

@app.post("/process-videos")
async def process_videos(files: List[UploadFile] = File(...)):
    """Process uploaded video files"""
    if not video_generator:
        raise HTTPException(status_code=500, detail="Video processor not available")
    
    results = []
    
    for file in files:
        start_time = time.time()
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp_file:
            tmp_path = tmp_file.name
            content = await file.read()
            tmp_file.write(content)
        
        try:
            # Process video
            result = video_generator.process_video_file(tmp_path)
            processing_time = time.time() - start_time
            
            results.append(ProcessingResult(
                filename=file.filename,
                success=result['success'],
                caption=result.get('caption', ''),
                transcript_preview=result.get('transcript', ''),
                error=result.get('error'),
                processing_time=processing_time
            ))
            
        except Exception as e:
            processing_time = time.time() - start_time
            results.append(ProcessingResult(
                filename=file.filename,
                success=False,
                caption='',
                transcript_preview='',
                error=str(e),
                processing_time=processing_time
            ))
        finally:
            # Cleanup temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    return {"results": [
        {
            "filename": r.filename,
            "success": r.success,
            "caption": r.caption,
            "transcript_preview": r.transcript_preview,
            "error": r.error,
            "processing_time": round(r.processing_time, 2)
        }
        for r in results
    ]}

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
        "status": "ok", 
        "message": "healthy",
        "video_processor": video_generator is not None,
        "youtube_processor": youtube_generator is not None,
        "timestamp": datetime.now().isoformat(),
        "platform": "Railway"
    }

if __name__ == "__main__":
    import uvicorn
    import os
    
    # Get port from environment or default to 8000
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 Starting server on port {port}")
    
    uvicorn.run(app, host="0.0.0.0", port=port)