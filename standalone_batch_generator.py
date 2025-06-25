#!/usr/bin/env python3
"""
Jasper's STANDALONE Batch YouTube Shorts Caption Generator 🚀
🎯 All-in-one file - no imports needed!

Process multiple scheduled Shorts URLs → Get all your captions ready!
"""

import os
import subprocess
import json
import csv
import time
import tempfile
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
from typing import List, Dict
from dataclasses import dataclass
from openai import OpenAI

@dataclass
class BatchResult:
    url: str
    success: bool
    caption: str
    transcript_preview: str
    error: str = None
    processing_time: float = 0

class YouTubeCaptionGenerator:
    def __init__(self, openai_api_key=None):
        """Initialize with OpenAI API key"""
        self.client = OpenAI(api_key=openai_api_key or os.getenv('OPENAI_API_KEY'))
        
        # Check if yt-dlp is installed
        try:
            subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("yt-dlp not found. Install with: pip install yt-dlp")
    
    def extract_transcript(self, youtube_url):
        """Extract transcript from YouTube Shorts"""
        print(f"🎥 Processing: {youtube_url}")
        
        # First, try to get auto-generated transcript
        transcript = self._get_auto_transcript(youtube_url)
        if transcript:
            print("✅ Found auto-generated transcript!")
            return transcript
        
        # Fallback: Download audio and transcribe
        print("⚡ No transcript found, downloading audio for transcription...")
        audio_path = self._download_audio(youtube_url)
        
        if audio_path:
            transcript = self._transcribe_audio(audio_path)
            # Cleanup temp file
            os.unlink(audio_path)
            return transcript
        
        return None
    
    def _get_auto_transcript(self, youtube_url):
        """Try to extract auto-generated captions using yt-dlp"""
        try:
            cmd = [
                'yt-dlp', '--write-auto-subs', '--write-subs', '--skip-download',
                '--sub-format', 'vtt', '--sub-langs', 'en', 
                youtube_url
            ]
            
            with tempfile.TemporaryDirectory() as temp_dir:
                result = subprocess.run(
                    cmd, 
                    cwd=temp_dir,
                    capture_output=True, 
                    text=True, 
                    timeout=30
                )
                
                # Look for .vtt files in temp directory
                vtt_files = list(Path(temp_dir).glob('*.vtt'))
                
                if vtt_files:
                    # Read the first VTT file and extract text
                    with open(vtt_files[0], 'r', encoding='utf-8') as f:
                        vtt_content = f.read()
                    
                    # Simple VTT parsing - extract text lines
                    lines = vtt_content.split('\n')
                    transcript_lines = []
                    
                    for line in lines:
                        line = line.strip()
                        # Skip timestamps and empty lines
                        if line and not line.startswith('WEBVTT') and '-->' not in line and not line.isdigit():
                            transcript_lines.append(line)
                    
                    return ' '.join(transcript_lines)
                    
        except subprocess.TimeoutExpired:
            print("⚠️ Transcript extraction timed out")
        except Exception as e:
            print(f"⚠️ Auto-transcript failed: {str(e)}")
        
        return None
    
    def _download_audio(self, youtube_url):
        """Download audio from YouTube video"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_path = temp_file.name
            
            cmd = [
                'yt-dlp', '-x', '--audio-format', 'mp3',
                '--audio-quality', '0',
                '-o', temp_path.replace('.mp3', '.%(ext)s'),
                youtube_url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                possible_files = [
                    temp_path,
                    temp_path.replace('.mp3', '.m4a'),
                    temp_path.replace('.mp3', '.webm')
                ]
                
                for file_path in possible_files:
                    if os.path.exists(file_path):
                        return file_path
            
        except subprocess.TimeoutExpired:
            print("⚠️ Audio download timed out")
        except Exception as e:
            print(f"⚠️ Audio download failed: {str(e)}")
        
        return None
    
    def _transcribe_audio(self, audio_path):
        """Transcribe audio using OpenAI Whisper"""
        try:
            print("🎤 Transcribing audio...")
            
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
    
    def generate_caption(self, transcript):
        """Generate catchy caption from transcript using GPT-4"""
        if not transcript:
            return "❌ Could not generate caption - no transcript available"
        
        print("✨ Generating caption...")
        
        prompt = f"""
You are a social media expert specializing in healthcare marketing. 

Your task: Create a catchy, engaging caption for a YouTube Short based on this transcript.

TRANSCRIPT:
{transcript}

REQUIREMENTS:
- 1-2 punchy sentences maximum
- Healthcare/medical tone but accessible to general audience  
- Include relevant emojis (2-3 max)
- Focus on the key insight or takeaway
- Make it shareable and engaging
- If it's educational content, lead with the main benefit/insight
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
            return caption
            
        except Exception as e:
            print(f"⚠️ Caption generation failed: {str(e)}")
            return f"❌ Caption generation error: {str(e)}"
    
    def process_shorts_url(self, youtube_url):
        """Main method - process YouTube Shorts URL and return caption"""
        try:
            # Validate URL
            if 'youtube.com/shorts/' not in youtube_url and 'youtu.be/' not in youtube_url:
                return {
                    'success': False,
                    'error': 'Invalid YouTube URL. Please provide a YouTube Shorts URL.'
                }
            
            # Extract transcript
            transcript = self.extract_transcript(youtube_url)
            
            if not transcript:
                return {
                    'success': False,
                    'error': 'Could not extract transcript or audio from video'
                }
            
            # Generate caption
            caption = self.generate_caption(transcript)
            
            return {
                'success': True,
                'url': youtube_url,
                'transcript': transcript[:200] + '...' if len(transcript) > 200 else transcript,
                'caption': caption
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Processing failed: {str(e)}'
            }

class BatchCaptionGenerator:
    def __init__(self, openai_api_key=None, max_workers=3):
        """Initialize batch processor"""
        self.generator = YouTubeCaptionGenerator(openai_api_key)
        self.max_workers = max_workers
        self.results = []
    
    def process_url_list(self, urls: List[str], progress_callback=None) -> List[BatchResult]:
        """Process multiple YouTube Shorts URLs in parallel"""
        print(f"🚀 Starting batch processing of {len(urls)} URLs...")
        print(f"⚡ Using {self.max_workers} parallel workers")
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_url = {
                executor.submit(self._process_single_url, url): url 
                for url in urls
            }
            
            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    # Progress update
                    print(f"✅ Completed {completed}/{len(urls)}: {url}")
                    if progress_callback:
                        progress_callback(completed, len(urls), result)
                        
                except Exception as e:
                    error_result = BatchResult(
                        url=url,
                        success=False,
                        caption="",
                        transcript_preview="",
                        error=str(e)
                    )
                    results.append(error_result)
                    completed += 1
                    print(f"❌ Failed {completed}/{len(urls)}: {url} - {str(e)}")
        
        self.results = results
        return results
    
    def _process_single_url(self, url: str) -> BatchResult:
        """Process a single URL and return BatchResult"""
        start_time = time.time()
        
        try:
            result = self.generator.process_shorts_url(url)
            processing_time = time.time() - start_time
            
            if result['success']:
                return BatchResult(
                    url=url,
                    success=True,
                    caption=result['caption'],
                    transcript_preview=result['transcript'],
                    processing_time=processing_time
                )
            else:
                return BatchResult(
                    url=url,
                    success=False,
                    caption="",
                    transcript_preview="",
                    error=result['error'],
                    processing_time=processing_time
                )
                
        except Exception as e:
            processing_time = time.time() - start_time
            return BatchResult(
                url=url,
                success=False,
                caption="",
                transcript_preview="",
                error=str(e),
                processing_time=processing_time
            )
    
    def save_results_csv(self, filename: str = None):
        """Save results to CSV file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"youtube_captions_batch_{timestamp}.csv"
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'url', 'success', 'caption', 'transcript_preview', 
                'error', 'processing_time_seconds'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in self.results:
                writer.writerow({
                    'url': result.url,
                    'success': result.success,
                    'caption': result.caption,
                    'transcript_preview': result.transcript_preview,
                    'error': result.error or '',
                    'processing_time_seconds': round(result.processing_time, 2)
                })
        
        print(f"💾 Results saved to: {filename}")
        return filename
    
    def print_summary(self):
        """Print batch processing summary"""
        if not self.results:
            print("No results to summarize!")
            return
        
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        total_time = sum(r.processing_time for r in self.results)
        
        print("\n" + "="*60)
        print("🎯 BATCH PROCESSING SUMMARY")
        print("="*60)
        print(f"📊 Total URLs: {len(self.results)}")
        print(f"✅ Successful: {len(successful)}")
        print(f"❌ Failed: {len(failed)}")
        print(f"⚡ Success Rate: {len(successful)/len(self.results)*100:.1f}%")
        print(f"⏱️ Total Time: {total_time:.1f}s")
        print(f"📈 Avg Time per URL: {total_time/len(self.results):.1f}s")
        
        if successful:
            print(f"\n🏆 SAMPLE CAPTIONS:")
            for i, result in enumerate(successful[:3]):
                print(f"{i+1}. {result.caption}")
        
        if failed:
            print(f"\n⚠️ FAILED URLS:")
            for result in failed:
                print(f"   • {result.url}: {result.error}")
        
        print("="*60)

def load_urls_from_file(file_path: str) -> List[str]:
    """Load URLs from text file or CSV"""
    urls = []
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if file_path.suffix.lower() == '.csv':
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            url_columns = [col for col in reader.fieldnames if 'url' in col.lower()]
            if not url_columns:
                raise ValueError("No URL column found in CSV")
            
            url_col = url_columns[0]
            for row in reader:
                if row[url_col].strip():
                    urls.append(row[url_col].strip())
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip()]
    
    return urls

def main():
    """CLI interface for batch processing"""
    parser = argparse.ArgumentParser(description='Batch process YouTube Shorts for captions')
    parser.add_argument('--urls', nargs='+', help='List of YouTube Shorts URLs')
    parser.add_argument('--file', help='File containing URLs (one per line or CSV)')
    parser.add_argument('--output-csv', help='Output CSV filename')
    parser.add_argument('--workers', type=int, default=3, help='Number of parallel workers')
    parser.add_argument('--api-key', help='OpenAI API Key')
    
    args = parser.parse_args()
    
    # Get URLs
    urls = []
    if args.urls:
        urls = args.urls
    elif args.file:
        urls = load_urls_from_file(args.file)
    else:
        print("❌ Please provide URLs via --urls or --file")
        return
    
    if not urls:
        print("❌ No URLs to process!")
        return
    
    print(f"🎬 Found {len(urls)} URLs to process")
    
    try:
        processor = BatchCaptionGenerator(
            openai_api_key=args.api_key,
            max_workers=args.workers
        )
        
        results = processor.process_url_list(urls)
        
        if args.output_csv:
            processor.save_results_csv(args.output_csv)
        else:
            processor.save_results_csv()
        
        processor.print_summary()
        
    except Exception as e:
        print(f"❌ Batch processing failed: {str(e)}")

if __name__ == "__main__":
    main()