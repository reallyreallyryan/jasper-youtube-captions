from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import List
import subprocess
from openai import OpenAI
import logging

# Enhanced logging for Railway debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YouTubeCaptionGenerator:
    def __init__(self, openai_api_key=None):
        self.client = OpenAI(api_key=openai_api_key or os.getenv('OPENAI_API_KEY'))
        
        # Enhanced yt-dlp checking with detailed logging
        self.yt_dlp_available = self._check_yt_dlp_installation()
    
    def _check_yt_dlp_installation(self):
        """Enhanced yt-dlp installation check with debugging"""
        try:
            logger.info("🔍 Checking yt-dlp installation...")
            
            # Check if yt-dlp executable exists
            result = subprocess.run(['which', 'yt-dlp'], capture_output=True, text=True)
            logger.info(f"yt-dlp location: {result.stdout.strip()}")
            
            # Check yt-dlp version
            version_result = subprocess.run(['yt-dlp', '--version'], capture_output=True, text=True, timeout=10)
            if version_result.returncode == 0:
                logger.info(f"✅ yt-dlp version: {version_result.stdout.strip()}")
                
                # Test a simple yt-dlp command
                test_result = subprocess.run([
                    'yt-dlp', '--help'
                ], capture_output=True, text=True, timeout=10)
                
                if test_result.returncode == 0:
                    logger.info("✅ yt-dlp help command works")
                    return True
                else:
                    logger.error(f"❌ yt-dlp help failed: {test_result.stderr}")
                    return False
            else:
                logger.error(f"❌ yt-dlp version check failed: {version_result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ yt-dlp check timed out")
            return False
        except FileNotFoundError:
            logger.error("❌ yt-dlp not found in PATH")
            return False
        except Exception as e:
            logger.error(f"❌ yt-dlp check failed: {str(e)}")
            return False
    
    def process_shorts_url(self, youtube_url):
        try:
            logger.info(f"🎥 Processing URL: {youtube_url}")
            
            if 'youtube.com/shorts/' not in youtube_url and 'youtu.be/' not in youtube_url:
                return {'success': False, 'error': 'Invalid YouTube URL'}
            
            if not self.yt_dlp_available:
                return {'success': False, 'error': 'yt-dlp not available on this platform'}
            
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
            logger.error(f"❌ Processing failed for {youtube_url}: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _extract_transcript(self, youtube_url):
        logger.info(f"📝 Extracting transcript from: {youtube_url}")
        
        # Try auto-transcript first
        transcript = self._get_auto_transcript(youtube_url)
        if transcript:
            logger.info("✅ Auto-transcript extracted successfully")
            return transcript
        
        logger.warning("❌ Auto-transcript extraction failed, trying audio download...")
        
        # Fallback: Download audio and transcribe
        audio_path = self._download_audio(youtube_url)
        if audio_path:
            transcript = self._transcribe_audio(audio_path)
            # Cleanup temp file
            if os.path.exists(audio_path):
                os.unlink(audio_path)
            return transcript
        
        logger.error("❌ Both auto-transcript and audio transcription failed")
        return None
    
    def _get_auto_transcript(self, youtube_url):
        """Enhanced auto-transcript extraction with detailed logging"""
        try:
            logger.info("🔍 Attempting auto-transcript extraction...")
            
            cmd = [
                'yt-dlp', 
                '--write-auto-subs', 
                '--write-subs', 
                '--skip-download',
                '--sub-format', 'vtt', 
                '--sub-langs', 'en',
                '--verbose',  # Add verbose logging
                youtube_url
            ]
            
            logger.info(f"🚀 Running command: {' '.join(cmd)}")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                logger.info(f"📁 Using temp directory: {temp_dir}")
                
                result = subprocess.run(
                    cmd, 
                    cwd=temp_dir, 
                    capture_output=True, 
                    text=True, 
                    timeout=60  # Increased timeout
                )
                
                logger.info(f"📊 yt-dlp return code: {result.returncode}")
                logger.info(f"📄 yt-dlp stdout: {result.stdout[:500]}...")  # First 500 chars
                
                if result.stderr:
                    logger.warning(f"⚠️ yt-dlp stderr: {result.stderr[:500]}...")
                
                # List all files in temp directory
                all_files = list(Path(temp_dir).glob('*'))
                logger.info(f"📂 Files in temp dir: {[f.name for f in all_files]}")
                
                vtt_files = list(Path(temp_dir).glob('*.vtt'))
                logger.info(f"🎬 VTT files found: {[f.name for f in vtt_files]}")
                
                if vtt_files:
                    vtt_file = vtt_files[0]
                    logger.info(f"📖 Reading VTT file: {vtt_file.name}")
                    
                    with open(vtt_file, 'r', encoding='utf-8') as f:
                        vtt_content = f.read()
                    
                    logger.info(f"📝 VTT content length: {len(vtt_content)} chars")
                    logger.info(f"📄 VTT preview: {vtt_content[:200]}...")
                    
                    # Simple VTT parsing - extract text lines
                    lines = vtt_content.split('\n')
                    transcript_lines = []
                    
                    for line in lines:
                        line = line.strip()
                        # Skip timestamps and empty lines
                        if line and not line.startswith('WEBVTT') and '-->' not in line and not line.isdigit():
                            transcript_lines.append(line)
                    
                    transcript = ' '.join(transcript_lines)
                    logger.info(f"✅ Extracted transcript: {len(transcript)} chars")
                    logger.info(f"📝 Transcript preview: {transcript[:100]}...")
                    
                    return transcript
                else:
                    logger.error("❌ No VTT files found")
                    return None
                    
        except subprocess.TimeoutExpired:
            logger.error("⏰ yt-dlp command timed out")
            return None
        except Exception as e:
            logger.error(f"💥 Auto-transcript extraction failed: {str(e)}")
            import traceback
            logger.error(f"🔍 Full traceback: {traceback.format_exc()}")
            return None
    
    def _download_audio(self, youtube_url):
        """Download audio from YouTube video"""
        try:
            logger.info("🎵 Downloading audio for transcription...")
            
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_path = temp_file.name
            
            cmd = [
                'yt-dlp', '-x', '--audio-format', 'mp3',
                '--audio-quality', '0',
                '-o', temp_path.replace('.mp3', '.%(ext)s'),
                youtube_url
            ]
            
            logger.info(f"🚀 Audio download command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            logger.info(f"📊 Audio download return code: {result.returncode}")
            if result.stdout:
                logger.info(f"📄 Audio stdout: {result.stdout[:300]}...")
            if result.stderr:
                logger.warning(f"⚠️ Audio stderr: {result.stderr[:300]}...")
            
            if result.returncode == 0:
                # Check for possible audio files
                possible_files = [
                    temp_path,
                    temp_path.replace('.mp3', '.m4a'),
                    temp_path.replace('.mp3', '.webm'),
                    temp_path.replace('.mp3', '.wav')
                ]
                
                for file_path in possible_files:
                    if os.path.exists(file_path):
                        logger.info(f"✅ Audio file created: {file_path}")
                        return file_path
                
                logger.error("❌ No audio file found despite success code")
                return None
            else:
                logger.error(f"❌ Audio download failed with code {result.returncode}")
                return None
            
        except subprocess.TimeoutExpired:
            logger.error("⏰ Audio download timed out")
            return None
        except Exception as e:
            logger.error(f"💥 Audio download failed: {str(e)}")
            return None
    
    def _transcribe_audio(self, audio_path):
        """Transcribe audio using OpenAI Whisper"""
        try:
            logger.info(f"🎤 Transcribing audio file: {audio_path}")
            
            # Check file exists and size
            if not os.path.exists(audio_path):
                logger.error("❌ Audio file doesn't exist")
                return None
                
            file_size = os.path.getsize(audio_path)
            logger.info(f"📁 Audio file size: {file_size} bytes")
            
            if file_size == 0:
                logger.error("❌ Audio file is empty")
                return None
            
            with open(audio_path, 'rb') as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            
            logger.info(f"✅ Whisper transcription complete: {len(transcript)} chars")
            logger.info(f"📝 Transcript preview: {transcript[:100]}...")
            return transcript
            
        except Exception as e:
            logger.error(f"💥 Whisper transcription failed: {str(e)}")
            return None
    
    def _generate_caption(self, transcript):
        if not transcript:
            return "❌ No transcript available"
        
        logger.info("✨ Generating caption with OpenAI...")
        
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
            
            caption = response.choices[0].message.content.strip()
            logger.info(f"✅ Generated caption: {caption}")
            return caption
            
        except Exception as e:
            logger.error(f"❌ Caption generation failed: {str(e)}")
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
        logger.info("🚀 Initializing YouTube caption generator...")
        youtube_generator = YouTubeCaptionGenerator()
        logger.info("✅ YouTube caption generator initialized!")
    except Exception as e:
        logger.error(f"💥 YouTube generator initialization failed: {e}")

@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse("static/index.html")

@app.get("/debug")
async def debug_info():
    """Debug endpoint to check system status"""
    debug_info = {
        "youtube_generator_available": youtube_generator is not None,
        "openai_api_key_set": bool(os.getenv('OPENAI_API_KEY')),
        "environment": os.getenv('RAILWAY_ENVIRONMENT', 'unknown'),
    }
    
    # Check yt-dlp manually
    try:
        result = subprocess.run(['yt-dlp', '--version'], capture_output=True, text=True, timeout=5)
        debug_info["yt_dlp_version"] = result.stdout.strip() if result.returncode == 0 else "failed"
        debug_info["yt_dlp_error"] = result.stderr if result.returncode != 0 else None
    except Exception as e:
        debug_info["yt_dlp_version"] = "not_found"
        debug_info["yt_dlp_error"] = str(e)
    
    # Check Python path and working directory
    debug_info["python_path"] = os.environ.get('PATH', '')
    debug_info["working_directory"] = os.getcwd()
    debug_info["temp_directory"] = tempfile.gettempdir()
    
    return debug_info

@app.post("/process-urls")
async def process_urls(urls: List[str]):
    """Process YouTube Shorts URLs"""
    logger.info(f"📥 Received {len(urls)} URLs to process")
    
    if not youtube_generator:
        logger.error("❌ YouTube processor not available")
        raise HTTPException(status_code=500, detail="YouTube processor not available")
    
    results = []
    
    for i, url in enumerate(urls, 1):
        logger.info(f"🔄 Processing URL {i}/{len(urls)}: {url}")
        start_time = time.time()
        
        try:
            result = youtube_generator.process_shorts_url(url.strip())
            processing_time = time.time() - start_time
            
            result_data = {
                "url": url,
                "success": result['success'],
                "caption": result.get('caption', ''),
                "transcript_preview": result.get('transcript', ''),
                "error": result.get('error'),
                "processing_time": round(processing_time, 2)
            }
            
            results.append(result_data)
            logger.info(f"✅ URL {i} processed: {'SUCCESS' if result['success'] else 'FAILED'}")
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"💥 URL {i} failed with exception: {str(e)}")
            
            results.append({
                "url": url,
                "success": False,
                "caption": '',
                "transcript_preview": '',
                "error": str(e),
                "processing_time": round(processing_time, 2)
            })
    
    logger.info(f"🎯 Batch processing complete: {len(results)} results")
    return {"results": results}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "youtube_processor": youtube_generator is not None,
        "yt_dlp_available": youtube_generator.yt_dlp_available if youtube_generator else False,
        "timestamp": datetime.now().isoformat(),
        "platform": "Railway",
        "features": ["YouTube URL processing"],
        "openai_configured": bool(os.getenv('OPENAI_API_KEY'))
    }

if __name__ == "__main__":
    import uvicorn
    import os
    
    port = int(os.getenv("PORT", 8000))
    logger.info(f"🚀 Starting server on port {port}")

    uvicorn.run(app, host="0.0.0.0", port=port)