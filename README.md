# Offline Translator

Translate between 200 languages entirely on your own computer. After a one-time
model download, no internet connection is needed and nothing you type leaves
your machine.

It uses Meta's **NLLB-200** model ("No Language Left Behind") and opens as a
web page in your browser at `http://127.0.0.1:8765`.

## Setup (one time, while online)

You need Python 3.9 or newer.

```bash
cd offline\_translator
python -m venv .venv
# Windows:      .venv\\Scripts\\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
python translator.py --download
```

The download is about 2.5 GB and is saved in the `models` folder next to the script.

## Use (works offline)

```bash
python translator.py
```

Your browser opens the translator. The first launch takes a little while as the
model loads into memory. The status badge turns green when it's ready.

* Pick **Detect language** or a specific source language, then the target language.
* Press **Translate** or Ctrl + Enter.
* **⇄** swaps the languages and moves the translation into the input box.
* Line breaks are kept. Long texts are split into sentences automatically.

## Options

|Option|What it does|
|-|-|
|`--model 1.3B`|Better quality model (\~5.5 GB). Download it first with `--download --model 1.3B`.|
|`--model 3.3B`|Best quality (\~17.6 GB). Needs a lot of RAM or a GPU.|
|`--device cuda`|Use an NVIDIA GPU (much faster). The default `auto` picks it when available.|
|`--beams 4`|Slightly better translations, slower. Default is 2.|
|`--port 9000`|Use a different port.|
|`--no-browser`|Don't open the browser automatically.|

## Good to know

* **Speed:** on a typical laptop CPU, a paragraph takes a few seconds with the 600M model.
* **Memory:** the 600M model needs roughly 3 to 4 GB of free RAM.
* **Romanized text** (for example Roman Urdu or Hinglish typed in Latin letters) doesn't
translate well. The model expects Urdu in Arabic script and Hindi in Devanagari.
* **Language detection** recognizes about 75 major languages. For anything else,
pick the source language yourself.
* **License:** the NLLB-200 models are released under CC-BY-NC 4.0, which means
**non-commercial use only**.
* The server only accepts connections from your own computer (127.0.0.1).

