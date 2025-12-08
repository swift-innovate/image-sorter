# 🖼️ Image Sorter

**AI-powered local image organization using BLIP captioning**

Automatically analyze, categorize, and rename images from your Downloads folder using a local AI model. No cloud APIs, no privacy concerns, runs entirely on your GPU.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.x-brightgreen.svg)

## ✨ Features

- **🧠 AI-Powered Captioning** - Uses Salesforce BLIP model to understand image content
- **🔒 Privacy-First** - Runs 100% locally, no data leaves your machine
- **⚡ GPU Accelerated** - ~0.2s per image on RTX 4080 (vs 60s+ on CPU)
- **📁 Smart Categorization** - Auto-sorts into Screenshots, Photos, AI_Generated, Memes, etc.
- **📝 Intelligent Renaming** - Replaces generic names (IMG_1234) with descriptive captions
- **📊 HTML Reports** - Beautiful reports for scheduled/automated runs

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
    keywords: [screenshot, screen, desktop, code, terminal]
    patterns: [Screenshot, Capture, Snip]
  Photos:
    keywords: [photo, portrait, landscape, nature]
    patterns: [IMG_, DSC, PXL_]
  AI_Generated:
    keywords: [digital art, fantasy, cyberpunk, concept art]
    patterns: [DALL, Midjourney, Stable, ComfyUI]
  # ... add your own categories
```

## 🖥️ System Requirements

- **Python** 3.10+
- **CUDA** 12.x compatible GPU (8GB+ VRAM recommended)
- **OS** Windows 10/11, Linux, macOS

### GPU Performance (approximate)
| GPU | Time per Image |
|-----|----------------|
| RTX 4080 | ~0.2s |
| RTX 3080 | ~0.3s |
| RTX 3060 | ~0.5s |
| CPU (fallback) | ~60s |

## 📦 Dependencies

**Core (auto mode):**
```
torch>=2.6.0
torchvision
transformers>=4.30.0
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
   - Program: `python`
   - Arguments: `G:\Projects\image-sorter\image_sorter.py`
   - Start in: `G:\Projects\image-sorter`
5. Finish

## 📊 Sample Report

Auto mode generates a beautiful HTML report:

- Summary stats (analyzed, moved, errors, duration)
- Category breakdown
- Full file listing with original → new names
- Error details for failed files

Reports are saved to your destination folder with timestamps:
`image_sorter_2025-12-08_143022.html`

## 🤝 Contributing

Contributions welcome! Please feel free to submit issues and PRs.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [Salesforce BLIP](https://github.com/salesforce/BLIP) for the image captioning model
- [Gradio](https://gradio.app/) for the review UI
- [Hugging Face](https://huggingface.co/) for model hosting

---

**Made with ☕ by [Swift Innovate](https://github.com/swift-innovate)**
