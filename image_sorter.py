"""
Image Sorter - AI-powered local image organization
Uses BLIP model for privacy-first image captioning and categorization

Copyright (c) 2025 Swift Innovate
MIT License - https://github.com/swift-innovate/image-sorter
"""

import os
import re
import shutil
import sys
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import hashlib
import time

import yaml
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# ============================================================================
# CONFIGURATION LOADER
# ============================================================================

DEFAULT_CONFIG = {
    "source_dir": "",
    "dest_dir": "",
    "page_size": 15,
    "max_filename_length": 120,
    "categories": {
        "Screenshots": {"keywords": ["screenshot", "screen", "code"], "patterns": ["Screenshot", "Capture"]},
        "Photos": {"keywords": ["photo", "portrait", "landscape"], "patterns": ["IMG_", "DSC"]},
        "AI_Generated": {"keywords": ["digital art", "fantasy", "cyberpunk"], "patterns": ["DALL", "Midjourney"]},
        "Misc": {"keywords": [], "patterns": []}
    },
    "generic_patterns": [
        r'^IMG[_-]?\d+',
        r'^Screenshot[_ ]?\d*',
        r'^\d+$',
    ]
}

def load_config() -> dict:
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent / "config.yaml"
    
    if not config_path.exists():
        print(f"⚠️  No config.yaml found at {config_path}")
        print("   Creating default config.yaml - please edit it with your paths!")
        save_default_config(config_path)
        return DEFAULT_CONFIG
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Validate required fields
        if not config.get("source_dir"):
            print("❌ Error: source_dir not set in config.yaml")
            sys.exit(1)
        if not config.get("dest_dir"):
            print("❌ Error: dest_dir not set in config.yaml")
            sys.exit(1)
            
        return config
    except Exception as e:
        print(f"❌ Error loading config.yaml: {e}")
        sys.exit(1)


def save_default_config(path: Path):
    """Save a default config file"""
    default_yaml = """# Image Sorter Configuration
# Edit these settings to customize the app

# Folder paths - CHANGE THESE TO YOUR PATHS
source_dir: ""  # e.g., "C:\\\\Users\\\\YourName\\\\Downloads"
dest_dir: ""    # e.g., "C:\\\\Users\\\\YourName\\\\Pictures\\\\Sorted"

# UI settings
page_size: 15
max_filename_length: 120

# Categories
categories:
  Screenshots:
    keywords: [screenshot, screen, desktop, window, browser, code, terminal]
    patterns: [Screenshot, Capture, Snip]
  AI_Generated:
    keywords: [digital art, fantasy, surreal, cyberpunk, sci-fi, concept art]
    patterns: [DALL, Midjourney, Stable, ComfyUI]
  Photos:
    keywords: [photo, photograph, portrait, landscape, nature, selfie]
    patterns: [IMG_, DSC, DCIM, PXL_]
  Memes:
    keywords: [meme, funny, comic, cartoon, reaction]
    patterns: [meme, reaction]
  Documents:
    keywords: [document, text, chart, graph, diagram]
    patterns: [doc, scan, pdf]
  Icons_Logos:
    keywords: [icon, logo, symbol, badge]
    patterns: [icon, logo, favicon]
  Wallpapers:
    keywords: [wallpaper, background, panorama]
    patterns: [wallpaper, background]
  Misc:
    keywords: []
    patterns: []

# Generic filename patterns (regex)
generic_patterns:
  - '^IMG[_-]?\\d+'
  - '^DSC[_-]?\\d+'
  - '^Screenshot[_ ]?\\d*'
  - '^PXL_\\d+'
  - '^\\d+$'
"""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(default_yaml)


# Load config at module level
CONFIG = load_config()

SOURCE_DIR = Path(CONFIG["source_dir"])
DEST_BASE = Path(CONFIG["dest_dir"])
PAGE_SIZE = CONFIG.get("page_size", 15)
MAX_FILENAME_LENGTH = CONFIG.get("max_filename_length", 120)
CATEGORIES = CONFIG.get("categories", DEFAULT_CONFIG["categories"])
CATEGORY_LIST = list(CATEGORIES.keys())
GENERIC_PATTERNS = CONFIG.get("generic_patterns", DEFAULT_CONFIG["generic_patterns"])

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff'}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ImageAnalysis:
    """Results from analyzing an image"""
    filepath: Path
    caption: str
    category: str
    suggested_name: str
    original_was_generic: bool
    width: int
    height: int
    file_size: int
    selected: bool = True
    error: Optional[str] = None
    
    
# ============================================================================
# MODEL LOADER (Singleton)
# ============================================================================

class ModelLoader:
    """Lazy-loads the BLIP model on first use"""
    _processor = None
    _model = None
    _device = None
    
    @classmethod
    def get_model(cls):
        if cls._model is None:
            print("🔄 Loading BLIP model (first run only)...")
            cls._device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"   Using device: {cls._device}")
            
            cls._processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base",
                use_fast=True
            )
            cls._model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            ).to(cls._device)
            
            print("✅ Model loaded!")
        return cls._processor, cls._model, cls._device


# ============================================================================
# FILENAME INTELLIGENCE
# ============================================================================

def is_generic_filename(filename: str) -> bool:
    """Determine if a filename is generic/auto-generated vs descriptive"""
    stem = Path(filename).stem
    
    for pattern in GENERIC_PATTERNS:
        if re.match(pattern, stem, re.IGNORECASE):
            return True
    
    letters_only = re.sub(r'[^a-zA-Z]', '', stem)
    if len(letters_only) < 4:
        return True
    
    if len(stem) < 5:
        return True
    
    if re.match(r'^\d{4}[-_]?\d{2}[-_]?\d{2}', stem):
        return True
        
    return False


def extract_meaningful_words(filename: str) -> str:
    """Extract meaningful words from an existing descriptive filename"""
    stem = Path(filename).stem
    words = re.sub(r'[-_]+', ' ', stem)
    words = re.sub(r'\s+[a-f0-9]{6,}$', '', words, flags=re.IGNORECASE)
    words = re.sub(r'\s+\d+$', '', words)
    words = re.sub(r'\s*\(\d+\)$', '', words)
    return words.strip()


def sanitize_filename(name: str) -> str:
    """Create a safe filename from text"""
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = re.sub(r'\s+', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_.')
    return name


def clean_caption(caption: str) -> str:
    """Clean up AI-generated caption for use in filename"""
    clean = caption.strip()
    prefixes_to_remove = [
        "a ", "an ", "the ", "there is ", "this is ", 
        "a picture of ", "an image of ", "a photo of ",
        "a photograph of ", "a screenshot of "
    ]
    for prefix in prefixes_to_remove:
        if clean.lower().startswith(prefix):
            clean = clean[len(prefix):]
    return clean


def generate_filename(caption: str, original_path: Path) -> tuple[str, bool]:
    """Generate a descriptive filename."""
    ext = original_path.suffix.lower()
    cleaned_caption = clean_caption(caption)
    safe_caption = sanitize_filename(cleaned_caption)
    file_hash = hashlib.md5(original_path.name.encode()).hexdigest()[:6]
    
    was_generic = is_generic_filename(original_path.name)
    
    if was_generic:
        new_name = f"{safe_caption}_{file_hash}"
    else:
        existing_words = extract_meaningful_words(original_path.name)
        safe_existing = sanitize_filename(existing_words)
        existing_lower = safe_existing.lower()
        caption_lower = safe_caption.lower()
        
        if caption_lower in existing_lower or existing_lower in caption_lower:
            new_name = f"{safe_existing}_{file_hash}"
        else:
            new_name = f"{safe_existing}_{safe_caption}_{file_hash}"
    
    max_stem_length = MAX_FILENAME_LENGTH - len(ext) - 1
    if len(new_name) > max_stem_length:
        new_name = new_name[:max_stem_length].rsplit('_', 1)[0]
        if not new_name.endswith(file_hash):
            trim_to = max_stem_length - len(file_hash) - 1
            new_name = new_name[:trim_to].rsplit('_', 1)[0] + '_' + file_hash
    
    return f"{new_name}{ext}", was_generic


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def get_image_files(source_dir: Path) -> list[Path]:
    """Find all image files in the source directory"""
    images = set()  # Use set to avoid duplicates on case-insensitive filesystems
    for ext in IMAGE_EXTENSIONS:
        images.update(source_dir.glob(f"*{ext}"))
        images.update(source_dir.glob(f"*{ext.upper()}"))
    return sorted(images, key=lambda p: p.stat().st_mtime, reverse=True)


def generate_caption(image: Image.Image) -> str:
    """Generate a caption for the image using BLIP"""
    processor, model, device = ModelLoader.get_model()
    
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize to BLIP's expected size - speedup for large images
    image.thumbnail((384, 384))
    
    inputs = processor(image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=20,
            num_beams=1,
            do_sample=False
        )
    
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


def classify_image(caption: str, filename: str, width: int, height: int) -> str:
    """Determine the best category for an image"""
    caption_lower = caption.lower()
    filename_lower = filename.lower()
    
    aspect_ratio = width / height if height > 0 else 1
    is_wide = aspect_ratio > 1.7 and width >= 1920
    
    scores = {}
    
    for category, criteria in CATEGORIES.items():
        if category == "Misc":
            continue
            
        score = 0
        keywords = criteria.get("keywords", [])
        patterns = criteria.get("patterns", [])
        
        for keyword in keywords:
            if keyword.lower() in caption_lower:
                score += 2
        
        for pattern in patterns:
            if pattern.lower() in filename_lower:
                score += 3
        
        if category == "Wallpapers" and is_wide:
            score += 2
        
        if category == "Screenshots":
            if any(word in caption_lower for word in ["showing", "displaying", "view of"]):
                score += 1
        
        scores[category] = score
    
    if scores:
        best_category = max(scores, key=scores.get)
        if scores[best_category] > 0:
            return best_category
    
    return "Misc"


def analyze_image(filepath: Path) -> ImageAnalysis:
    """Perform full analysis on a single image"""
    try:
        img = Image.open(filepath)
        width, height = img.size
        file_size = filepath.stat().st_size
        
        if hasattr(img, 'n_frames') and img.n_frames > 1:
            img.seek(0)
        
        caption = generate_caption(img)
        
        # Close file handle explicitly
        img.close()
        
        category = classify_image(caption, filepath.name, width, height)
        suggested_name, was_generic = generate_filename(caption, filepath)
        
        return ImageAnalysis(
            filepath=filepath,
            caption=caption,
            category=category,
            suggested_name=suggested_name,
            original_was_generic=was_generic,
            width=width,
            height=height,
            file_size=file_size
        )
    except Exception as e:
        print(f"❌ Error analyzing {filepath}: {e}")
        return ImageAnalysis(
            filepath=filepath,
            caption="",
            category="Misc",
            suggested_name=filepath.name,
            original_was_generic=False,
            width=0,
            height=0,
            file_size=0,
            selected=False,
            error=str(e)
        )


def move_and_rename(analysis: ImageAnalysis, new_name: str, new_category: str) -> tuple[bool, str]:
    """Copy image to destination, verify, then delete original"""
    try:
        t_start = time.time()
        
        final_name = new_name if new_name else analysis.suggested_name
        final_category = new_category if new_category else analysis.category
        
        if not Path(final_name).suffix:
            final_name += analysis.filepath.suffix
        
        # Create destination directory
        dest_dir = DEST_BASE / final_category
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Handle duplicates
        dest_path = dest_dir / final_name
        counter = 1
        while dest_path.exists():
            stem = Path(final_name).stem
            ext = Path(final_name).suffix
            dest_path = dest_dir / f"{stem}_{counter}{ext}"
            counter += 1
        
        source_path = analysis.filepath
        source_size = source_path.stat().st_size
        
        print(f"    📤 Source: {source_path}")
        print(f"    📥 Dest:   {dest_path}")
        print(f"    📏 Size:   {source_size:,} bytes")
        
        # COPY first (don't move)
        t_copy_start = time.time()
        shutil.copy2(str(source_path), str(dest_path))
        t_copy_end = time.time()
        print(f"    ⏱️  Copy took: {t_copy_end - t_copy_start:.2f}s")
        
        # VERIFY the copy
        if not dest_path.exists():
            print(f"    ❌ FAILED: Destination file doesn't exist after copy!")
            return False, f"❌ Copy failed - dest not found"
        
        dest_size = dest_path.stat().st_size
        if dest_size != source_size:
            print(f"    ❌ FAILED: Size mismatch! Source={source_size}, Dest={dest_size}")
            dest_path.unlink()  # Remove bad copy
            return False, f"❌ Copy failed - size mismatch"
        
        print(f"    ✅ Verified: {dest_size:,} bytes")
        
        # Only delete source AFTER successful verification
        t_delete_start = time.time()
        source_path.unlink()
        t_delete_end = time.time()
        print(f"    🗑️  Deleted original ({t_delete_end - t_delete_start:.2f}s)")
        print(f"    ✅ Total: {t_delete_end - t_start:.2f}s")
        
        return True, f"✅ {dest_path.name}"
    except Exception as e:
        print(f"    ❌ ERROR: {e}")
        return False, f"❌ {e}"


# ============================================================================
# BATCH PROCESSOR APP
# ============================================================================

class BatchImageSorter:
    """Batch processing UI for image sorting (--review mode)"""
    
    def __init__(self):
        self.all_analyses: list[ImageAnalysis] = []
        self.current_page = 0
        
    def get_total_pages(self) -> int:
        if not self.all_analyses:
            return 1
        return (len(self.all_analyses) + PAGE_SIZE - 1) // PAGE_SIZE
    
    def get_page_data(self) -> list[ImageAnalysis]:
        start = self.current_page * PAGE_SIZE
        end = start + PAGE_SIZE
        return self.all_analyses[start:end]
    
    def build_table_data(self) -> list[list]:
        """Build table data for current page"""
        page_data = self.get_page_data()
        rows = []
        
        for analysis in page_data:
            status = "⚠️ Error" if analysis.error else ("🔄 Generic" if analysis.original_was_generic else "✨ Enhanced")
            size_kb = analysis.file_size / 1024 if analysis.file_size else 0
            
            rows.append([
                analysis.selected,
                analysis.filepath.name[:40] + "..." if len(analysis.filepath.name) > 40 else analysis.filepath.name,
                analysis.suggested_name[:50] + "..." if len(analysis.suggested_name) > 50 else analysis.suggested_name,
                analysis.category,
                status,
                f"{size_kb:.0f}KB",
                analysis.caption[:60] + "..." if len(analysis.caption) > 60 else analysis.caption
            ])
        
        return rows
    
    def update_from_table(self, table_data) -> str:
        """Update analyses from edited table data"""
        # Handle DataFrame or empty data
        if table_data is None:
            return "No data"
        
        # Convert DataFrame to list if needed
        if hasattr(table_data, 'values'):
            table_data = table_data.values.tolist()
        
        if not table_data or len(table_data) == 0:
            return "No data"
        
        page_data = self.get_page_data()
        
        for i, row in enumerate(table_data):
            if i < len(page_data) and len(row) >= 4:
                page_data[i].selected = bool(row[0])
                # Update suggested name if changed (handle truncation)
                if row[2] and not str(row[2]).endswith("..."):
                    page_data[i].suggested_name = str(row[2])
                if row[3]:
                    page_data[i].category = str(row[3])
        
        selected = sum(1 for a in self.all_analyses if a.selected)
        return f"Selected: {selected}/{len(self.all_analyses)}"
    
    def approve_selected(self) -> tuple[str, list[list]]:
        """Move all selected images"""
        selected = [a for a in self.all_analyses if a.selected and not a.error]
        
        if not selected:
            return "No images selected", self.build_table_data()
        
        success_count = 0
        fail_count = 0
        
        print(f"\n📦 Moving {len(selected)} files...")
        t_batch_start = time.time()
        
        for i, analysis in enumerate(selected):
            print(f"\n  [{i + 1}/{len(selected)}]")
            success, msg = move_and_rename(analysis, analysis.suggested_name, analysis.category)
            if success:
                success_count += 1
                self.all_analyses.remove(analysis)
            else:
                fail_count += 1
                analysis.error = msg
        
        t_batch_end = time.time()
        print(f"\n📦 Batch complete: {success_count} moved, {fail_count} failed in {t_batch_end - t_batch_start:.2f}s")
        
        if self.current_page >= self.get_total_pages():
            self.current_page = max(0, self.get_total_pages() - 1)
        
        return f"✅ Moved {success_count}, {fail_count} failed. {len(self.all_analyses)} remaining.", self.build_table_data()
    
    def select_all_page(self) -> list[list]:
        """Select all on current page"""
        for analysis in self.get_page_data():
            if not analysis.error:
                analysis.selected = True
        return self.build_table_data()
    
    def deselect_all_page(self) -> list[list]:
        """Deselect all on current page"""
        for analysis in self.get_page_data():
            analysis.selected = False
        return self.build_table_data()
    
    def select_all(self) -> list[list]:
        """Select ALL images"""
        for analysis in self.all_analyses:
            if not analysis.error:
                analysis.selected = True
        return self.build_table_data()
    
    def next_page(self) -> tuple[str, list[list]]:
        """Go to next page"""
        if self.current_page < self.get_total_pages() - 1:
            self.current_page += 1
        return f"Page {self.current_page + 1} of {self.get_total_pages()}", self.build_table_data()
    
    def prev_page(self) -> tuple[str, list[list]]:
        """Go to previous page"""
        if self.current_page > 0:
            self.current_page -= 1
        return f"Page {self.current_page + 1} of {self.get_total_pages()}", self.build_table_data()
    
    def get_preview_image(self, evt) -> Optional[Image.Image]:
        """Get preview image when row is clicked"""
        if evt.index[0] is not None:
            page_data = self.get_page_data()
            if evt.index[0] < len(page_data):
                try:
                    img = Image.open(page_data[evt.index[0]].filepath)
                    img.thumbnail((500, 500))
                    return img
                except:
                    pass
        return None
    
    def build_ui(self, gr):
        """Build the batch processing Gradio interface"""
        
        total = len(self.all_analyses)
        errors = sum(1 for a in self.all_analyses if a.error)
        
        with gr.Blocks(title="🖼️ Image Sorter v2") as app:
            gr.Markdown("# 🖼️ Image Sorter v2\n*Review and organize your images*")
            gr.Markdown(f"**Source:** `{SOURCE_DIR}`  →  **Destination:** `{DEST_BASE}`")
            gr.Markdown(f"✅ **{total} images analyzed** ({errors} errors) - Ready for review!")
            
            with gr.Row():
                move_status = gr.Textbox(label="Status", value="Select images and click Approve & Move", interactive=False)
            
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        prev_btn = gr.Button("◀ Previous")
                        page_label = gr.Textbox(value="Page 1 of 1", label="", interactive=False, scale=1)
                        next_btn = gr.Button("Next ▶")
                    
                    table = gr.Dataframe(
                        headers=["✓", "Original Name", "New Name", "Category", "Status", "Size", "Caption"],
                        datatype=["bool", "str", "str", "str", "str", "str", "str"],
                        col_count=(7, "fixed"),
                        row_count=(PAGE_SIZE, "fixed"),
                        interactive=True,
                        label="Images (click to preview, edit New Name and Category)"
                    )
                    
                    with gr.Row():
                        select_all_page_btn = gr.Button("☑️ Select Page")
                        deselect_btn = gr.Button("☐ Deselect Page")
                        select_all_btn = gr.Button("☑️ Select ALL", variant="secondary")
                    
                    with gr.Row():
                        approve_btn = gr.Button("✅ Approve & Move Selected", variant="primary", size="lg")
                        selection_status = gr.Textbox(label="Selection", interactive=False)
                
                with gr.Column(scale=1):
                    preview_image = gr.Image(label="Preview (click row to view)", height=400)
                    gr.Markdown("### Categories\n" + 
                               "\n".join([f"- {cat}" for cat in CATEGORY_LIST]))
            
            # Event handlers
            table.select(
                fn=self.get_preview_image,
                outputs=[preview_image]
            )
            
            prev_btn.click(fn=self.prev_page, outputs=[page_label, table])
            next_btn.click(fn=self.next_page, outputs=[page_label, table])
            select_all_page_btn.click(fn=self.select_all_page, outputs=[table])
            deselect_btn.click(fn=self.deselect_all_page, outputs=[table])
            select_all_btn.click(fn=self.select_all, outputs=[table])
            
            approve_btn.click(
                fn=self.approve_selected,
                outputs=[move_status, table]
            ).then(
                fn=lambda: f"Page {self.current_page + 1} of {self.get_total_pages()}",
                outputs=[page_label]
            )
            
            # Load initial data on startup
            app.load(
                fn=lambda: (self.build_table_data(), f"Page 1 of {self.get_total_pages()}"),
                outputs=[table, page_label]
            )
        
        return app


# ============================================================================
# HTML REPORT GENERATOR
# ============================================================================

def generate_html_report(
    results: list[dict],
    start_time: datetime,
    end_time: datetime,
    total_analyzed: int,
    errors: int
) -> str:
    """Generate an HTML report of the sorting operation"""
    
    duration = (end_time - start_time).total_seconds()
    moved_count = sum(1 for r in results if r.get('success'))
    failed_count = sum(1 for r in results if not r.get('success'))
    
    # Count by category
    category_counts = {}
    for r in results:
        if r.get('success'):
            cat = r.get('category', 'Unknown')
            category_counts[cat] = category_counts.get(cat, 0) + 1
    
    # Build category breakdown HTML
    category_html = ""
    for cat in sorted(category_counts.keys()):
        count = category_counts[cat]
        category_html += f"<tr><td>{cat}</td><td>{count}</td></tr>\n"
    
    # Build results table HTML
    results_html = ""
    for r in results:
        status = "✅" if r.get('success') else "❌"
        error_msg = r.get('error', '')
        row_class = "success" if r.get('success') else "error"
        results_html += f"""<tr class="{row_class}">
            <td>{status}</td>
            <td title="{r.get('original', '')}">{r.get('original', '')[:50]}</td>
            <td title="{r.get('new_name', '')}">{r.get('new_name', '')[:50]}</td>
            <td>{r.get('category', '')}</td>
            <td title="{r.get('caption', '')}">{r.get('caption', '')[:60]}</td>
            <td>{error_msg}</td>
        </tr>\n"""
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Sorter Report - {start_time.strftime('%Y-%m-%d')}</title>
    <style>
        :root {{
            --bg: #1a1a2e;
            --card: #16213e;
            --accent: #e94560;
            --success: #00d26a;
            --error: #ff6b6b;
            --text: #eee;
            --muted: #888;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: var(--bg);
            color: var(--text);
            padding: 2rem;
            line-height: 1.6;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ color: var(--accent); margin-bottom: 0.5rem; }}
        .subtitle {{ color: var(--muted); margin-bottom: 2rem; }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .stat-card {{
            background: var(--card);
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
        }}
        .stat-value {{ font-size: 2rem; font-weight: bold; color: var(--accent); }}
        .stat-label {{ color: var(--muted); font-size: 0.9rem; }}
        .section {{ background: var(--card); border-radius: 12px; padding: 1.5rem; margin-bottom: 2rem; }}
        .section h2 {{ color: var(--accent); margin-bottom: 1rem; font-size: 1.2rem; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
        th, td {{ padding: 0.75rem; text-align: left; border-bottom: 1px solid #333; }}
        th {{ color: var(--accent); font-weight: 600; }}
        tr.success td:first-child {{ color: var(--success); }}
        tr.error {{ background: rgba(255, 107, 107, 0.1); }}
        tr.error td:first-child {{ color: var(--error); }}
        td {{ max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
        .paths {{ color: var(--muted); font-size: 0.85rem; margin-bottom: 1rem; }}
        .paths code {{ background: #333; padding: 0.2rem 0.5rem; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🖼️ Image Sorter Report</h1>
        <p class="subtitle">{start_time.strftime('%B %d, %Y at %I:%M %p')}</p>
        
        <div class="paths">
            <strong>Source:</strong> <code>{SOURCE_DIR}</code><br>
            <strong>Destination:</strong> <code>{DEST_BASE}</code>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">{total_analyzed}</div>
                <div class="stat-label">Analyzed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="color: var(--success)">{moved_count}</div>
                <div class="stat-label">Moved</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="color: var(--error)">{failed_count + errors}</div>
                <div class="stat-label">Errors</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{duration:.1f}s</div>
                <div class="stat-label">Duration</div>
            </div>
        </div>
        
        <div class="section">
            <h2>📊 Category Breakdown</h2>
            <table>
                <thead><tr><th>Category</th><th>Count</th></tr></thead>
                <tbody>{category_html}</tbody>
            </table>
        </div>
        
        <div class="section">
            <h2>📄 All Results ({len(results)} files)</h2>
            <table>
                <thead>
                    <tr>
                        <th></th>
                        <th>Original</th>
                        <th>New Name</th>
                        <th>Category</th>
                        <th>Caption</th>
                        <th>Error</th>
                    </tr>
                </thead>
                <tbody>{results_html}</tbody>
            </table>
        </div>
        
        <p style="color: var(--muted); text-align: center; margin-top: 2rem;">
            Generated by Image Sorter v2.1<br>
            &copy; 2025 Swift Innovate • MIT License • 
            <a href="https://github.com/swift-innovate/image-sorter" style="color: var(--accent);">GitHub</a>
        </p>
    </div>
</body>
</html>"""
    
    return html


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="AI-powered image sorter using local BLIP model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python image_sorter.py              # Auto-move all, generate report
  python image_sorter.py --review     # Interactive UI for manual review
  python image_sorter.py --dry-run    # Analyze only, no moves
"""
    )
    parser.add_argument("--review", action="store_true", 
                        help="Launch interactive UI for manual review")
    parser.add_argument("--dry-run", action="store_true",
                        help="Analyze only, don't move any files")
    args = parser.parse_args()
    
    mode = "Review" if args.review else "Dry Run" if args.dry_run else "Auto"
    
    print("=" * 60)
    print(f"🖼️  IMAGE SORTER v2.1 - {mode} Mode")
    print("=" * 60)
    print(f"Source:      {SOURCE_DIR}")
    print(f"Destination: {DEST_BASE}")
    print(f"Config:      {Path(__file__).parent / 'config.yaml'}")
    print("=" * 60)
    
    DEST_BASE.mkdir(parents=True, exist_ok=True)
    start_time = datetime.now()
    
    # =========================================
    # PHASE 1: Analyze all images
    # =========================================
    print("\n🧠 Loading BLIP model...")
    ModelLoader.get_model()
    print("✅ Model loaded!\n")
    
    image_files = get_image_files(SOURCE_DIR)
    total = len(image_files)
    
    if total == 0:
        print("❌ No images found in source folder!")
        return
    
    print(f"🔄 Analyzing {total} images...")
    print("-" * 40)
    
    all_analyses = []
    total_time = 0
    
    for i, filepath in enumerate(image_files):
        t_start = time.time()
        analysis = analyze_image(filepath)
        t_elapsed = time.time() - t_start
        total_time += t_elapsed
        all_analyses.append(analysis)
        
        avg_time = total_time / (i + 1)
        eta = avg_time * (total - i - 1)
        
        status = "⚠️" if analysis.error else "✅"
        print(f"  {status} [{i + 1}/{total}] {filepath.name[:40]} ({t_elapsed:.2f}s) ETA: {eta:.0f}s")
    
    analysis_errors = sum(1 for a in all_analyses if a.error)
    print("-" * 40)
    print(f"✅ Analysis complete! {total} images, {analysis_errors} errors, {total_time:.1f}s")
    print(f"   Average: {total_time/total:.2f}s per image")
    
    # =========================================
    # PHASE 2: Mode-specific handling
    # =========================================
    
    if args.dry_run:
        # Dry run - just show what would happen
        print("\n📋 DRY RUN - No files moved")
        print("\nCategory breakdown:")
        categories = {}
        for a in all_analyses:
            if not a.error:
                categories[a.category] = categories.get(a.category, 0) + 1
        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count}")
        return
    
    if args.review:
        # Review mode - launch Gradio UI (lazy import)
        try:
            import gradio as gr
        except ImportError:
            print("❌ Gradio not installed. Install with: pip install gradio")
            print("   Or use auto mode (default) which doesn't require Gradio.")
            return
        
        print("\n🚀 Launching review UI...")
        
        app_instance = BatchImageSorter()
        app_instance.all_analyses = all_analyses
        
        ui = app_instance.build_ui(gr)
        ui.launch(inbrowser=True)
        return
    
    # Default: Auto mode - move everything, generate report
    print("\n📦 Moving images...")
    print("-" * 40)
    
    results = []
    moveable = [a for a in all_analyses if not a.error]
    
    for i, analysis in enumerate(moveable):
        success, msg = move_and_rename(analysis, analysis.suggested_name, analysis.category)
        
        results.append({
            'success': success,
            'original': analysis.filepath.name,
            'new_name': analysis.suggested_name,
            'category': analysis.category,
            'caption': analysis.caption,
            'error': '' if success else msg
        })
        
        status = "✅" if success else "❌"
        print(f"  {status} [{i + 1}/{len(moveable)}] {analysis.filepath.name[:40]}")
    
    # Add analysis errors to results
    for a in all_analyses:
        if a.error:
            results.append({
                'success': False,
                'original': a.filepath.name,
                'new_name': '',
                'category': '',
                'caption': '',
                'error': a.error
            })
    
    end_time = datetime.now()
    
    # Generate HTML report
    print("\n📊 Generating report...")
    html = generate_html_report(results, start_time, end_time, total, analysis_errors)
    
    report_name = f"image_sorter_{start_time.strftime('%Y-%m-%d_%H%M%S')}.html"
    report_path = DEST_BASE / report_name
    report_path.write_text(html, encoding='utf-8')
    
    moved = sum(1 for r in results if r['success'])
    failed = sum(1 for r in results if not r['success'])
    duration = (end_time - start_time).total_seconds()
    
    print("-" * 40)
    print(f"✅ Complete! {moved} moved, {failed} errors, {duration:.1f}s")
    print(f"📄 Report: {report_path}")


if __name__ == "__main__":
    main()
