> [!IMPORTANT]
> **Help Wanted – Reverse Engineering**
>
> I’m looking for anyone who can help reverse engineer the game’s geometry format (`.geo`) files.
> I’ve tried cracking it many times with no success so far.
>
> Any help would be **greatly appreciated**.

# ZOMBI-Manager

**ZOMBI-Manager** is a Python application for browsing and extracting files from **ZOMBI(U)** archives.  
It also includes a **work-in-progress importer** intended to eventually support modding the game.

The main focus of the tool is exploring `.bfz` archive files and inspecting their contents.


> [!WARNING]
> In its current state, the importer is currently broken, as it just loses too much data when trying to convert it to a `.bfz` file.
>
> It will be reworked in the future to actually function correctly (it will ask you to pick a 'base' file to use as a start to import the new / replaced data.

---

## What this tool can do

- Open and browse `.bfz` archive files
- View contained files in:
  - Hex view
  - String view
- Play back `.son` audio files
- Export:
  - Individual files (raw or converted)
  - Entire archives to a folder

---

## Understanding `.bfz` files

If you need help figuring out what different `.bfz` files contain:

- See **`WOR Descriptions`** in the root of the repository
- This file lists:
  - The WOR / `.bfz` name
  - Its original path during game compilation

The paths were sourced from debug logs found in the game (via TCRF), and can help you make educated guesses about each archive’s contents.

---

## Getting Started

### Requirements

- **Python 3**
- The following Python dependencies (I may have missed one or two — it’s been a while):
  - `PySide6`
  - `python-lzo`
  - `Pillow`
  - `numpy`

### Running the app

1. Download or clone the repository
2. Navigate to the `ZOMBIManager` folder
3. Run:

   ```bash
   python zombiManager.py
