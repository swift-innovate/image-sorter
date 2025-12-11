# 🖼️ Image Sorter

**AI-powered local image organization using Qwen2.5-VL**

Automatically analyze, categorize, and rename images from your Downloads folder using a local vision-language AI model. No cloud APIs, no privacy concerns, runs entirely on your GPU.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.x-brightgreen.svg)

## ✨ Features

- **🧠 High-Quality Captions** - Qwen2.5-VL-3B for detailed, context-aware descriptions
- **🔒 Privacy-First** - Runs 100% locally, no data leaves your machine
- **⚡ GPU Accelerated** - ~0.5s per image on RTX 4080
- **📁 Smart Categorization** - Auto-sorts into Screenshots, Photos, AI_Generated, Memes, etc.
- **📝 Intelligent Renaming** - Replaces generic names (IMG_1234) with descriptive captions
- **📊 HTML Reports** - Beautiful reports for scheduled/automated runs
- **🔄 Auto-Fallback** - Falls back to BLIP on GPUs with <7GB VRAM

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/swift-innovate/image-sorter.git
cd image-sorter

# Install dependencies
pip install -r requirements.txt

# Copy and edit config
cp config.yaml.example config.yaml
# Edit config.yaml with your source/destination paths

# Run it (auto-moves all images, generates report)
python image_sorter.py

# Or use review mode for manual approval
python image_sorter.py --review
```

## 📋 Usage Modes

### Auto Mode (default)
```bash
python image_sorter.py
```
- Analyzes all images in source folder
- Moves them to categorized subfolders
- Generates HTML report in destination
- Perfect for scheduled tasks

### Review Mode
```bash
python image_sorter.py --review
```
- Opens web UI after analysis
- Preview images, adjust categories, edit names
- Approve and move selected images

### Dry Run Mode
```bash
python image_sorter.py --dry-run
```
- Analyzes images without moving anything
- Shows category breakdown
- Test your config before committing

## 🧠 Model Selection

Image Sorter automatically selects the best model for your hardware:

| VRAM | Model | Caption Quality |
|------|-------|-----------------|
| 7GB+ | Qwen2.5-VL-3B | "screenshot of VS Code with Python error in dark terminal" |
| <7GB | BLIP (fallback) | "text on a computer screen" |

No configuration needed - it just works!

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
# Folder paths
source_dir: "C:\\Users\\YourName\\Downloads"
dest_dir: "C:\\Users\\YourName\\Pictures\\Sorted"

# UI settings
page_size: 15
max_filename_length: 120

# Categories with keywords and filename patterns
categories:
  Screenshots:
    keywords: [screenshot, screen, desktop, code, terminal, interface]
    patterns: [Screenshot, Capture, Snip]
  Photos:
    keywords: [photo, portrait, landscape, nature, outdoor]
    patterns: [IMG_, DSC, PXL_]
  AI_Generated:
    keywords: [digital art, fantasy, cyberpunk, concept art, rendered]
    patterns: [DALL, Midjourney, Stable, ComfyUI]
  # ... add your own categories
```

## 🖥️ System Requirements

- **Python** 3.10+
- **CUDA** 12.x compatible GPU
- **VRAM** 7GB+ recommended (falls back to BLIP on less)
- **OS** Windows 10/11, Linux, macOS

### GPU Performance (approximate)
| GPU | VRAM | Model | Speed |
|-----|------|-------|-------|
| RTX 4080 | 16GB | Qwen-3B | ~0.5s |
| RTX 4070 | 12GB | Qwen-3B | ~0.6s |
| RTX 3060 | 8GB | Qwen-3B | ~0.8s |
| GTX 1660 | 6GB | BLIP | ~0.3s |
| CPU | - | BLIP | ~30-60s |

## 📦 Dependencies

**Core:**
```
torch>=2.6.0
torchvision
transformers>=4.45.0
qwen-vl-utils>=0.0.8
accelerate>=0.27.0
Pillow>=9.0.0
pyyaml>=6.0
```

**Optional (for --review mode):**
```bash
pip install gradio>=4.0.0
```

## 🔧 Windows Task Scheduler Setup

1. Open Task Scheduler
2. Create Basic Task → Name: "Image Sorter Weekly"
3. Trigger: Weekly (or your preference)
4. Action: Start a program
   - Program: `G:\Projects\image-sorter\scheduled_run.bat`
5. Finish

## 📊 Sample Report

Auto mode generates a beautiful HTML report showing:
- Summary stats (analyzed, moved, errors, duration)
- Model used (Qwen or BLIP)
- Category breakdown
- Full file listing with original → new names

Reports saved to destination folder: `image_sorter_2025-12-08_143022.html`

## 🤝 Contributing

Contributions welcome! See [TODO.md](TODO.md) for improvement ideas.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2-VL) - Primary vision-language model
- [Salesforce BLIP](https://github.com/salesforce/BLIP) - Lightweight fallback model
- [Gradio](https://gradio.app/) - Review UI framework
- [Hugging Face](https://huggingface.co/) - Model hosting

---

**Made with ☕ by [Swift Innovate](https://github.com/swift-innovate)**
