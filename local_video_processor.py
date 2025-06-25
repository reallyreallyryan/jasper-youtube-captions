#!/usr/bin/env python3
"""
Jasper's Local Video Caption Processor 🎬
Process video files locally → Generate captions BEFORE uploading anywhere!

Perfect for:
- Pre-upload workflow
- Draft video processing
- Any video format (MP4, MOV, AVI, etc.)
- Batch processing entire folders
"""

import os
import subprocess
import tempfile
import json
import csv
from datetime import datetime
from pathlib import Path
import argparse
from typing import List, Dict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

@dataclass
class VideoResult:
    file_path: str
    success: bool
    caption: str
    transcript_preview: str
    error: str = None
    processing_time: float = 0

class LocalVideoCaptionGenerator:
    def __init__(self, openai_api_key=None):
        """Initialize local video processor"""
        self.client = OpenAI(api_key=openai_api_key or os.getenv('OPENAI_API_KEY'))
        
        # Check if ffmpeg is installed
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("ffmpeg not found. Install with: brew install ffmpeg (Mac) or apt install ffmpeg (Linux)")
    
    def process_video_file(self, video_path):
        """Process a single video file and generate caption"""
        print(f"🎬 Processing: {video_path}")
        
        try:
            # Extract audio from video
            audio_path = self._extract_audio(video_path)
            
            if not audio_path:
                return {
                    'success': False,
                    'error': 'Could not extract audio from video'
                }
            
            # Transcribe audio
            transcript = self._transcribe_audio(audio_path)
            
            # Cleanup temp audio file
            if os.path.exists(audio_path):
                os.unlink(audio_path)
            
            if not transcript:
                return {
                    'success': False,
                    'error': 'Could not transcribe audio'
                }
            
            # Generate caption
            caption = self._generate_caption(transcript)
            
            return {
                'success': True,
                'caption': caption,
                'transcript': transcript[:200] + '...' if len(transcript) > 200 else transcript
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Processing failed: {str(e)}'
            }
    
    def _extract_audio(self, video_path):
        """Extract audio from video file using ffmpeg"""
        try:
            # Create temp file for audio
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                audio_path = temp_file.name
            
            # Extract audio using ffmpeg
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vn',  # No video
                '-acodec', 'mp3',  # Audio codec
                '-ar', '16000',  # Sample rate (good for Whisper)
                '-ac', '1',  # Mono audio
                '-y',  # Overwrite output file
                audio_path
            ]
            
            print("🎵 Extracting audio...")
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=120  # 2 minute timeout
            )
            
            if result.returncode == 0 and os.path.exists(audio_path):
                print("✅ Audio extracted successfully!")
                return audio_path
            else:
                print(f"❌ Audio extraction failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print("⚠️ Audio extraction timed out")
            return None
        except Exception as e:
            print(f"⚠️ Audio extraction error: {str(e)}")
            return None
    
    def _transcribe_audio(self, audio_path):
        """Transcribe audio using OpenAI Whisper"""
        try:
            print("🎤 Transcribing audio with Whisper...")
            
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
        """Generate healthcare marketing caption from transcript"""
        if not transcript:
            return "❌ No transcript available for caption generation"
        
        print("✨ Generating caption...")
        
        prompt = f"""
You are a social media expert specializing in healthcare marketing. 

Your task: Create a catchy, engaging caption for a video based on this transcript.

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
            print("✅ Caption generated!")
            return caption
            
        except Exception as e:
            print(f"⚠️ Caption generation failed: {str(e)}")
            return f"❌ Caption error: {str(e)}"

class BatchVideoProcessor:
    def __init__(self, openai_api_key=None, max_workers=2):
        """Initialize batch processor (fewer workers for video processing)"""
        self.generator = LocalVideoCaptionGenerator(openai_api_key)
        self.max_workers = max_workers
        self.results = []
    
    def process_video_files(self, video_paths):
        """Process multiple video files"""
        print(f"🚀 Starting batch processing of {len(video_paths)} videos...")
        print(f"⚡ Using {self.max_workers} parallel workers")
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self._process_single_video, path): path 
                for path in video_paths
            }
            
            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_path):
                video_path = future_to_path[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    status = "✅ SUCCESS" if result.success else "❌ FAILED"
                    print(f"{status} {completed}/{len(video_paths)}: {Path(video_path).name}")
                    
                except Exception as e:
                    error_result = VideoResult(
                        file_path=video_path,
                        success=False,
                        caption="",
                        transcript_preview="",
                        error=str(e)
                    )
                    results.append(error_result)
                    completed += 1
                    print(f"❌ ERROR {completed}/{len(video_paths)}: {Path(video_path).name} - {str(e)}")
        
        self.results = results
        return results
    
    def _process_single_video(self, video_path):
        """Process single video and return VideoResult"""
        import time
        start_time = time.time()
        
        try:
            result = self.generator.process_video_file(video_path)
            processing_time = time.time() - start_time
            
            return VideoResult(
                file_path=video_path,
                success=result['success'],
                caption=result.get('caption', ''),
                transcript_preview=result.get('transcript', ''),
                error=result.get('error'),
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return VideoResult(
                file_path=video_path,
                success=False,
                caption="",
                transcript_preview="",
                error=str(e),
                processing_time=processing_time
            )
    
    def save_results_csv(self, filename=None):
        """Save results to CSV"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"local_video_captions_{timestamp}.csv"
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'video_file', 'success', 'caption', 'transcript_preview', 
                'error', 'processing_time_seconds'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in self.results:
                writer.writerow({
                    'video_file': Path(result.file_path).name,
                    'success': result.success,
                    'caption': result.caption,
                    'transcript_preview': result.transcript_preview,
                    'error': result.error or '',
                    'processing_time_seconds': round(result.processing_time, 2)
                })
        
        print(f"💾 Results saved to: {filename}")
        return filename
    
    def print_summary(self):
        """Print processing summary"""
        if not self.results:
            print("No results to summarize!")
            return
        
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        total_time = sum(r.processing_time for r in self.results)
        
        print("\n" + "="*60)
        print("🎯 LOCAL VIDEO PROCESSING SUMMARY")
        print("="*60)
        print(f"📊 Total Videos: {len(self.results)}")
        print(f"✅ Successful: {len(successful)}")
        print(f"❌ Failed: {len(failed)}")
        print(f"⚡ Success Rate: {len(successful)/len(self.results)*100:.1f}%")
        print(f"⏱️ Total Time: {total_time:.1f}s")
        print(f"📈 Avg Time per Video: {total_time/len(self.results):.1f}s")
        
        if successful:
            print(f"\n🏆 SAMPLE CAPTIONS:")
            for i, result in enumerate(successful[:3]):
                video_name = Path(result.file_path).name
                print(f"{i+1}. {video_name}: {result.caption}")
        
        if failed:
            print(f"\n⚠️ FAILED VIDEOS:")
            for result in failed:
                video_name = Path(result.file_path).name
                print(f"   • {video_name}: {result.error}")
        
        print("="*60)

def find_video_files(directory):
    """Find all video files in directory"""
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(Path(directory).glob(f'*{ext}'))
        video_files.extend(Path(directory).glob(f'*{ext.upper()}'))
    
    return [str(f) for f in video_files]

def main():
    """CLI interface"""
    parser = argparse.ArgumentParser(description='Process local video files to generate captions')
    parser.add_argument('input', help='Video file or directory containing videos')
    parser.add_argument('--output', help='Output CSV filename')
    parser.add_argument('--workers', type=int, default=2, help='Number of parallel workers (default: 2)')
    parser.add_argument('--api-key', help='OpenAI API Key')
    
    args = parser.parse_args()
    
    # Determine input type
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single video file
        video_files = [str(input_path)]
    elif input_path.is_dir():
        # Directory of videos
        video_files = find_video_files(input_path)
        if not video_files:
            print(f"❌ No video files found in {input_path}")
            return
    else:
        print(f"❌ Input path not found: {input_path}")
        return
    
    print(f"🎬 Found {len(video_files)} video file(s) to process")
    
    try:
        if len(video_files) == 1:
            # Single file processing
            generator = LocalVideoCaptionGenerator(args.api_key)
            result = generator.process_video_file(video_files[0])
            
            if result['success']:
                print(f"\n🎉 SUCCESS!")
                print(f"📹 Video: {Path(video_files[0]).name}")
                print(f"✨ Caption: {result['caption']}")
                print(f"📝 Transcript: {result['transcript']}")
            else:
                print(f"\n❌ FAILED!")
                print(f"Error: {result['error']}")
        else:
            # Batch processing
            processor = BatchVideoProcessor(
                openai_api_key=args.api_key,
                max_workers=args.workers
            )
            
            results = processor.process_video_files(video_files)
            
            if args.output:
                processor.save_results_csv(args.output)
            else:
                processor.save_results_csv()
            
            processor.print_summary()
        
    except Exception as e:
        print(f"❌ Processing failed: {str(e)}")

if __name__ == "__main__":
    main()