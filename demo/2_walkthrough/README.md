# 2 · Walkthrough

A page in your browser that steps through a video one model call at a time: what
the model sees, what it replies, what the pipeline does with the reply, and what
is saved in `trace.json`.

```bash
python demo/2_walkthrough/walkthrough.py --video clip.mp4 --task st
```

Then open `http://127.0.0.1:8765` (it opens by itself). `walkthrough.py` is the
server; `web/` is the page. Details: [the demo README](../README.md#2--walkthrough).
