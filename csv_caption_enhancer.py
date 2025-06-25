#!/usr/bin/env python3
"""
Jasper's CSV Caption Enhancer 🔥
Takes your content calendar CSV → Adds AI-generated captions!

Input:  CSV with YouTube URLs
Output: Same CSV + caption columns filled out

Perfect for:
- Content calendar automation
- Scheduled Shorts workflow
- Client deliverables that look PROFESSIONAL AF
"""

import os
import csv
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
from typing import List, Dict
from dataclasses import dataclass
from openai import OpenAI
import subprocess
import tempfile
import json

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
        # First, try to get auto-generated transcript
        transcript = self._get_auto_transcript(youtube_url)
        if transcript:
            return transcript
        
        # Fallback: Download audio and transcribe
        audio_path = self._download_audio(youtube_url)
        if audio_path:
            transcript = self._transcribe_audio(audio_path)
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
                    
        except Exception:
            pass
        
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
            
        except Exception:
            pass
        
        return None
    
    def _transcribe_audio(self, audio_path):
        """Transcribe audio using OpenAI Whisper"""
        try:
            with open(audio_path, 'rb') as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            
            return transcript
            
        except Exception:
            return None
    
    def generate_caption(self, transcript):
        """Generate catchy caption from transcript using GPT-4"""
        if not transcript:
            return "❌ No transcript available"
        
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
    
    def process_shorts_url(self, youtube_url):
        """Process a single URL"""
        transcript = self.extract_transcript(youtube_url)
        if not transcript:
            return {
                'success': False,
                'caption': '❌ Could not extract transcript',
                'transcript': ''
            }
        
        caption = self.generate_caption(transcript)
        return {
            'success': True,
            'caption': caption,
            'transcript': transcript[:100] + '...' if len(transcript) > 100 else transcript
        }

class CSVCaptionEnhancer:
    def __init__(self, openai_api_key=None, max_workers=3):
        """Initialize CSV enhancer"""
        self.generator = YouTubeCaptionGenerator(openai_api_key)
        self.max_workers = max_workers
    
    def enhance_csv(self, input_csv_path, output_csv_path=None, url_column=None):
        """
        Take CSV with URLs → Add caption columns
        
        Args:
            input_csv_path: Path to input CSV
            output_csv_path: Path to save enhanced CSV (auto-generated if None)
            url_column: Name of URL column (auto-detected if None)
        """
        print(f"📊 Processing CSV: {input_csv_path}")
        
        # Read the CSV
        rows, headers, url_col = self._read_csv(input_csv_path, url_column)
        
        print(f"🎯 Found {len(rows)} rows with URL column: '{url_col}'")
        print(f"⚡ Using {self.max_workers} parallel workers")
        
        # Process URLs in parallel
        enhanced_rows = self._process_rows_parallel(rows, url_col)
        
        # Add new columns to headers
        new_headers = headers + ['ai_caption', 'ai_transcript_preview', 'ai_status']
        
        # Generate output filename if not provided
        if not output_csv_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            input_name = Path(input_csv_path).stem
            output_csv_path = f"{input_name}_with_captions_{timestamp}.csv"
        
        # Write enhanced CSV
        self._write_csv(enhanced_rows, new_headers, output_csv_path)
        
        # Print summary
        self._print_summary(enhanced_rows)
        
        return output_csv_path
    
    def _read_csv(self, csv_path, url_column=None):
        """Read CSV and find URL column"""
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            rows = list(reader)
        
        # Find URL column
        if url_column:
            if url_column not in headers:
                raise ValueError(f"Column '{url_column}' not found in CSV")
            url_col = url_column
        else:
            # Auto-detect URL column
            url_columns = [col for col in headers if 'url' in col.lower()]
            if not url_columns:
                raise ValueError("No URL column found. Specify with --url-column")
            url_col = url_columns[0]
        
        return rows, headers, url_col
    
    def _process_rows_parallel(self, rows, url_column):
        """Process all rows with URLs in parallel"""
        enhanced_rows = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_row = {
                executor.submit(self._process_single_row, row, url_column): (i, row) 
                for i, row in enumerate(rows)
            }
            
            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_row):
                row_index, original_row = future_to_row[future]
                
                try:
                    enhanced_row = future.result()
                    enhanced_rows.append((row_index, enhanced_row))
                    completed += 1
                    
                    url = original_row.get(url_column, 'N/A')
                    status = enhanced_row.get('ai_status', 'unknown')
                    print(f"✅ {completed}/{len(rows)}: {status} - {url}")
                    
                except Exception as e:
                    # Create error row
                    error_row = original_row.copy()
                    error_row.update({
                        'ai_caption': f'❌ Processing error: {str(e)}',
                        'ai_transcript_preview': '',
                        'ai_status': 'error'
                    })
                    enhanced_rows.append((row_index, error_row))
                    completed += 1
                    print(f"❌ {completed}/{len(rows)}: error - {str(e)}")
        
        # Sort by original order
        enhanced_rows.sort(key=lambda x: x[0])
        return [row for _, row in enhanced_rows]
    
    def _process_single_row(self, row, url_column):
        """Process a single CSV row"""
        enhanced_row = row.copy()
        url = row.get(url_column, '').strip()
        
        if not url or 'youtube.com/shorts/' not in url:
            enhanced_row.update({
                'ai_caption': '❌ Invalid or missing YouTube Shorts URL',
                'ai_transcript_preview': '',
                'ai_status': 'invalid_url'
            })
            return enhanced_row
        
        # Process the URL
        result = self.generator.process_shorts_url(url)
        
        if result['success']:
            enhanced_row.update({
                'ai_caption': result['caption'],
                'ai_transcript_preview': result['transcript'],
                'ai_status': 'success'
            })
        else:
            enhanced_row.update({
                'ai_caption': result['caption'],
                'ai_transcript_preview': '',
                'ai_status': 'failed'
            })
        
        return enhanced_row
    
    def _write_csv(self, rows, headers, output_path):
        """Write enhanced CSV"""
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"💾 Enhanced CSV saved to: {output_path}")
    
    def _print_summary(self, rows):
        """Print processing summary"""
        total = len(rows)
        successful = len([r for r in rows if r.get('ai_status') == 'success'])
        failed = len([r for r in rows if r.get('ai_status') in ['failed', 'error']])
        invalid = len([r for r in rows if r.get('ai_status') == 'invalid_url'])
        
        print("\n" + "="*60)
        print("🎯 CSV ENHANCEMENT SUMMARY")
        print("="*60)
        print(f"📊 Total Rows: {total}")
        print(f"✅ Successful Captions: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"⚠️ Invalid URLs: {invalid}")
        print(f"⚡ Success Rate: {successful/total*100:.1f}%")
        
        # Show sample captions
        successful_rows = [r for r in rows if r.get('ai_status') == 'success']
        if successful_rows:
            print(f"\n🏆 SAMPLE CAPTIONS:")
            for i, row in enumerate(successful_rows[:3]):
                print(f"{i+1}. {row['ai_caption']}")
        
        print("="*60)

def main():
    """CLI interface"""
    parser = argparse.ArgumentParser(description='Enhance CSV with AI-generated captions')
    parser.add_argument('input_csv', help='Input CSV file with YouTube URLs')
    parser.add_argument('--output', help='Output CSV filename (auto-generated if not provided)')
    parser.add_argument('--url-column', help='Name of URL column (auto-detected if not provided)')
    parser.add_argument('--workers', type=int, default=3, help='Number of parallel workers')
    parser.add_argument('--api-key', help='OpenAI API Key')
    
    args = parser.parse_args()
    
    try:
        enhancer = CSVCaptionEnhancer(
            openai_api_key=args.api_key,
            max_workers=args.workers
        )
        
        output_file = enhancer.enhance_csv(
            input_csv_path=args.input_csv,
            output_csv_path=args.output,
            url_column=args.url_column
        )
        
        print(f"\n🎉 SUCCESS! Enhanced CSV ready: {output_file}")
        
    except Exception as e:
        print(f"❌ Enhancement failed: {str(e)}")

if __name__ == "__main__":
    main()