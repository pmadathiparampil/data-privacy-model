#!/usr/bin/env python3
"""
Offline Translator: 200 languages, running entirely on your own computer.

Uses Meta's NLLB-200 model ("No Language Left Behind") through Hugging Face
transformers, and serves a small web page at http://127.0.0.1:8765

  1) One time, while online:    python translator.py --download
  2) Any time after, offline:   python translator.py

Run `python translator.py --help` for options (bigger models, port, GPU, ...).
"""

import argparse
import json
import os
import re
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"

# Pick with --model. All three support the same 200 languages.
MODELS = {
    "600M": "facebook/nllb-200-distilled-600M",  # ~2.5 GB download, fine on most laptops
    "1.3B": "facebook/nllb-200-distilled-1.3B",  # ~5.5 GB, noticeably better, slower
    "3.3B": "facebook/nllb-200-3.3B",            # ~17.6 GB, best quality, needs lots of RAM or a GPU
}

LANG_CODE = re.compile(r"^[a-z]{3}_[A-Z][a-z]{3}$")  # NLLB codes look like eng_Latn, urd_Arab
MAX_BODY = 200_000       # max request size in bytes
MAX_CHUNK_CHARS = 400    # longer paragraphs are split into sentences before translating
BATCH_SIZE = 8

# Languages written without spaces between words (chunks are re-joined without a space).
NO_SPACE = {"zho_Hans", "zho_Hant", "yue_Hant", "jpn_Jpan", "tha_Thai",
            "lao_Laoo", "khm_Khmr", "mya_Mymr", "bod_Tibt", "dzo_Tibt"}

# Shown at the top of the language menus.
PINNED = ["eng_Latn", "spa_Latn", "fra_Latn", "deu_Latn", "por_Latn", "ita_Latn",
          "rus_Cyrl", "arb_Arab", "hin_Deva", "urd_Arab", "ben_Beng", "pan_Guru",
          "pes_Arab", "tur_Latn", "zho_Hans", "jpn_Jpan", "kor_Hang", "vie_Latn",
          "ind_Latn", "tgl_Latn"]

# The language detector reports ISO 639-3 codes; these need a specific NLLB variant.
DETECT_TO_NLLB = {
    "ara": "arb_Arab", "zho": "zho_Hans", "msa": "zsm_Latn", "fas": "pes_Arab",
    "lav": "lvs_Latn", "nor": "nob_Latn", "nob": "nob_Latn", "nno": "nno_Latn",
    "aze": "azj_Latn", "swa": "swh_Latn", "mon": "khk_Cyrl", "uzb": "uzn_Latn",
    "sqi": "als_Latn", "est": "est_Latn", "kur": "kmr_Latn", "pus": "pbt_Arab",
}


# --------------------------------------------------------------------------- text helpers

SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|(?<=[。！？؟۔।॥])")


def _hard_split(text, limit):
    """Split text that is still too long, on spaces, or by characters if it has none."""
    if len(text) <= limit:
        return [text]
    parts, current = [], ""
    for word in text.split(" "):
        while len(word) > limit:  # e.g. Chinese/Japanese with no spaces at all
            if current:
                parts.append(current)
                current = ""
            parts.append(word[:limit])
            word = word[limit:]
        if current and len(current) + 1 + len(word) > limit:
            parts.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        parts.append(current)
    return parts


def _sentences(text):
    """Split into sentences, keeping each sentence's original trailing spacing."""
    parts, start = [], 0
    for match in SENTENCE_END.finditer(text):
        parts.append(text[start:match.end()])
        start = match.end()
    parts.append(text[start:])
    return [p for p in parts if p]


def split_chunks(text, limit=MAX_CHUNK_CHARS):
    """Break one paragraph into model-sized chunks, preferring sentence boundaries."""
    if len(text) <= limit:
        return [text]
    chunks, current = [], ""
    for sentence in _sentences(text):
        if len(sentence) > limit:  # one very long sentence: cut it on its own
            if current.strip():
                chunks.append(current.strip())
            current = ""
            chunks.extend(_hard_split(sentence.strip(), limit))
        elif current and len(current) + len(sentence) > limit:
            chunks.append(current.strip())
            current = sentence
        else:
            current += sentence
    if current.strip():
        chunks.append(current.strip())
    return chunks


def language_names(codes):
    """Readable names for NLLB codes, e.g. urd_Arab -> 'Urdu'. Falls back to the code."""
    try:
        import langcodes
    except ImportError:
        return {c: c for c in codes}
    per_language = {}
    for c in codes:
        per_language[c.split("_")[0]] = per_language.get(c.split("_")[0], 0) + 1
    names = {}
    for c in codes:
        lang, script = c.split("_")
        try:
            # Add the script only when a language appears in more than one script.
            tag = f"{lang}-{script}" if per_language[lang] > 1 else lang
            name = langcodes.Language.get(tag).display_name()
            for suffix in (" (individual language)", " (macrolanguage)"):
                name = name.replace(suffix, "")
            if not name or name.lower().startswith("unknown"):
                name = c
        except Exception:
            name = c
        names[c] = name
    return names


# --------------------------------------------------------------------------- translator

class Translator:
    def __init__(self, model_dir: Path, model_key: str, device: str, beams: int):
        self.model_dir = model_dir
        self.model_key = model_key
        self.device_pref = device
        self.beams = beams
        self.device = None
        self.status = "loading"
        self.error = None
        self.tokenizer = None
        self.model = None
        self.languages = []          # [{"code": "eng_Latn", "name": "English"}, ...]
        self.codes = set()
        self._generate_lock = threading.Lock()
        self._detector_lock = threading.Lock()
        self._detector = None
        self._detector_tried = False

    # ---- loading
    def load_tokenizer(self):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), local_files_only=True)
        codes = sorted(t for t in self.tokenizer.get_vocab() if LANG_CODE.match(t))
        if not codes:  # fallback: the list built into transformers
            from transformers.models.nllb.tokenization_nllb import FAIRSEQ_LANGUAGE_CODES
            codes = sorted(FAIRSEQ_LANGUAGE_CODES)
        self.set_languages(codes)

    def set_languages(self, codes):
        names = language_names(codes)
        self.codes = set(codes)
        self.languages = sorted(({"code": c, "name": names[c]} for c in codes),
                                key=lambda item: item["name"].lower())

    def load_model(self):
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM
            device = self.device_pref
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            model = AutoModelForSeq2SeqLM.from_pretrained(
                str(self.model_dir), local_files_only=True, torch_dtype=dtype)
            model.to(device).eval()
            self.model, self.device, self.status = model, device, "ready"
            self.detector()  # warm up language detection in the background
        except Exception as exc:  # shown in the page
            self.status, self.error = "error", f"{type(exc).__name__}: {exc}"

    def status_dict(self):
        return {"status": self.status, "error": self.error,
                "model": self.model_key, "device": self.device,
                "detect_available": self.detector_available()}

    # ---- language detection (optional, needs lingua-language-detector)
    def detector_available(self):
        try:
            import lingua  # noqa: F401
            return True
        except ImportError:
            return False

    def detector(self):
        with self._detector_lock:
            if not self._detector_tried:
                self._detector_tried = True
                try:
                    from lingua import LanguageDetectorBuilder
                    self._detector = LanguageDetectorBuilder.from_all_languages().build()
                except Exception:
                    self._detector = None
            return self._detector

    def name_of(self, code):
        for item in self.languages:
            if item["code"] == code:
                return item["name"]
        return code

    def _iso_to_code(self, iso):
        if iso in DETECT_TO_NLLB and DETECT_TO_NLLB[iso] in self.codes:
            return DETECT_TO_NLLB[iso]
        matches = sorted(c for c in self.codes if c.startswith(iso + "_"))
        return matches[0] if matches else None

    def detect(self, text):
        detector = self.detector()
        if detector is None:
            return None
        values = detector.compute_language_confidence_values(text[:1000])
        if not values:
            return None
        top = values[0]
        code = self._iso_to_code(top.language.iso_code_639_3.name.lower())
        if not code:
            return None
        return {"code": code, "name": self.name_of(code), "confidence": round(float(top.value), 3)}

    # ---- translation
    def translate(self, text, src, tgt):
        if src == tgt:
            return text
        import torch
        tok, model = self.tokenizer, self.model
        lines = text.split("\n")
        pieces = []  # (line index, chunk of text)
        for i, line in enumerate(lines):
            if line.strip():
                pieces.extend((i, chunk) for chunk in split_chunks(line.strip()))

        outputs = []
        target_id = tok.convert_tokens_to_ids(tgt)
        with self._generate_lock, torch.inference_mode():
            tok.src_lang = src
            for start in range(0, len(pieces), BATCH_SIZE):
                batch = [chunk for _, chunk in pieces[start:start + BATCH_SIZE]]
                enc = tok(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=512).to(self.device)
                max_new = min(512, int(enc["input_ids"].shape[1] * 2) + 16)
                generated = model.generate(**enc, forced_bos_token_id=target_id,
                                           max_new_tokens=max_new, num_beams=self.beams)
                outputs.extend(tok.batch_decode(generated, skip_special_tokens=True))

        joined = {}
        for (i, _), out in zip(pieces, outputs):
            joined.setdefault(i, []).append(out.strip())
        separator = "" if tgt in NO_SPACE else " "
        result = [separator.join(joined[i]) if i in joined else "" for i in range(len(lines))]
        return "\n".join(result)


# --------------------------------------------------------------------------- web server

def make_handler(translator, index_html: bytes):
    class Handler(BaseHTTPRequestHandler):
        server_version = "OfflineTranslator/1.0"

        def log_message(self, fmt, *args):  # keep the console quiet
            pass

        def _send(self, code, body, content_type="application/json; charset=utf-8"):
            data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _local_only(self):
            host = self.headers.get("Host", "").rsplit(":", 1)[0].strip("[]")
            if host not in ("127.0.0.1", "localhost", "::1"):
                self._send(403, {"error": "Only local access is allowed."})
                return False
            return True

        def do_GET(self):
            if not self._local_only():
                return
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                return self._send(200, index_html, "text/html; charset=utf-8")
            if path == "/api/languages":
                pinned = [c for c in PINNED if c in translator.codes]
                return self._send(200, {"languages": translator.languages, "pinned": pinned,
                                        "detect_available": translator.detector_available()})
            if path == "/api/status":
                return self._send(200, translator.status_dict())
            self._send(404, {"error": "Not found."})

        def do_POST(self):
            if not self._local_only():
                return
            path = self.path.split("?", 1)[0]
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                return self._send(413, {"error": "That text is too long for one request. Split it up."})
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                return self._send(400, {"error": "Invalid request."})
            text = str(payload.get("text", ""))

            if path == "/api/detect":
                found = translator.detect(text) if text.strip() else None
                return self._send(200, {"detected": found})

            if path != "/api/translate":
                return self._send(404, {"error": "Not found."})
            if translator.status != "ready":
                message = ("The model is still loading. Try again in a moment."
                           if translator.status == "loading" else f"The model failed to load: {translator.error}")
                return self._send(503, {"error": message})
            if not text.strip():
                return self._send(200, {"translation": "", "source": None, "detected": None, "seconds": 0})

            source, target = payload.get("source"), payload.get("target")
            detected = None
            if source == "auto":
                detected = translator.detect(text)
                if not detected:
                    return self._send(422, {"error": "Couldn't detect the language. Pick the source language instead."})
                source = detected["code"]
            if source not in translator.codes or target not in translator.codes:
                return self._send(400, {"error": "Unknown language."})

            started = time.time()
            try:
                translation = translator.translate(text, source, target)
            except Exception as exc:
                return self._send(500, {"error": f"Translation failed: {exc}"})
            self._send(200, {"translation": translation, "source": source, "detected": detected,
                             "seconds": round(time.time() - started, 2)})

    return Handler


# --------------------------------------------------------------------------- commands

def model_dir_for(model_key):
    return MODELS_DIR / MODELS[model_key].split("/", 1)[1]


def download(model_key):
    from huggingface_hub import snapshot_download
    repo = MODELS[model_key]
    target = model_dir_for(model_key)
    print(f"Downloading {repo} to {target}\nThis is a one-time download of several GB.")
    snapshot_download(repo_id=repo, local_dir=str(target),
                      ignore_patterns=["*.msgpack", "*.h5", "flax_*", "tf_*", "*.onnx", "rust_model.ot"])
    print("\nDone. You can now go offline and start the translator with:\n"
          f"  python translator.py{'' if model_key == '600M' else ' --model ' + model_key}")


def serve(args):
    # Never reach out to the internet while running.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    model_dir = model_dir_for(args.model)
    if not (model_dir / "config.json").exists():
        print(f"The {args.model} model isn't downloaded yet.\n"
              f"While online, run:  python translator.py --download"
              f"{'' if args.model == '600M' else ' --model ' + args.model}")
        sys.exit(1)

    translator = Translator(model_dir, args.model, args.device, args.beams)
    print("Reading language list...")
    translator.load_tokenizer()
    print(f"{len(translator.languages)} languages available. Loading the model in the background...")
    threading.Thread(target=translator.load_model, daemon=True).start()

    index_html = (HERE / "index.html").read_bytes()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(translator, index_html))
    url = f"http://127.0.0.1:{args.port}"
    print(f"Offline Translator is running at {url}   (press Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="Offline translator for 200 languages (NLLB-200).")
    parser.add_argument("--download", action="store_true",
                        help="download the model (run once while online)")
    parser.add_argument("--model", choices=MODELS, default="600M",
                        help="model size: 600M (default), 1.3B or 3.3B")
    parser.add_argument("--port", type=int, default=8765, help="port for the web page (default 8765)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="run on CPU or an NVIDIA GPU (default: auto)")
    parser.add_argument("--beams", type=int, default=2,
                        help="beam search width: higher is a bit better but slower (default 2)")
    parser.add_argument("--no-browser", action="store_true", help="don't open the browser automatically")
    args = parser.parse_args()
    if args.download:
        download(args.model)
    else:
        serve(args)


if __name__ == "__main__":
    main()
