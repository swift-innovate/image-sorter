# TODO / Future Considerations

Ideas for future improvements. PRs welcome!

---

## 1. Make it import-friendly

Currently `CONFIG = load_config()` runs at import time and can call `sys.exit(1)` if config is missing/invalid. Fine for script-only use, but problematic for:
- Importing from another script
- Shipping as a package

**Fix:** Move config loading into `main()` or a lazy `get_config()` function.

---

## 2. Recursive scanning

`get_image_files()` only scans the top level of `source_dir`. For Downloads folders this is fine, but photo archives often need recursive scanning.

**Fix:** Add config flag:
```yaml
recursive: false  # set true to scan subfolders
```
Then use `rglob` when enabled.

---

## 3. Resource handling

`analyze_image` closes the image after captioning, but if `generate_caption` throws, the file handle can leak.

**Fix:** Use context manager:
```python
with Image.open(filepath) as img:
    # work here
```

---

## 4. Batch captioning (performance)

Currently BLIP runs once per image. For large batches (hundreds/thousands), significant speedup possible by:
- Building small batches of tensors (8-16 images)
- Running `model.generate` once per batch

Optional optimization for a future "v3 performance mode."

---

## 5. HTML report polish

Current report is functional but could be more polished:
- Add `file://` links for dest paths (Windows local viewing)
- Table striping/borders in the inline CSS
- Maybe a dark/light theme toggle

---

## 6. Review mode: Error filtering

Errors are marked with "⚠️ Error" in the table, but no easy way to filter.

**Fix:** Add an "Errors only" filter button in Gradio UI to quickly review what failed (corrupt images, permissions, etc.).

---

## 7. PyInstaller / standalone exe

For non-technical users, a standalone `.exe` would be nice:
- Bundle Python + dependencies
- First-run config wizard
- Potential Gumroad release ($7-12)

---

## Contributing

Pick any item above and submit a PR! Please keep the privacy-first, local-only philosophy intact.
