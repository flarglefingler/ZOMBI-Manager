# ZOMBI-Manager

**ZOMBI-Manager** is a Python tool for browsing, inspecting, and extracting files from **ZOMBI / ZombiU** game archives.

The main focus right now is `.bfz` archives: opening them, looking through their contents, and exporting files for research, preservation, or modding experiments.

> [!WARNING]
> Archive importing/repacking is currently broken.
>
> The current importer loses too much data when trying to rebuild `.bfz` files, so do not use it for serious modding yet.
>
> I plan to rework it around a “base file” flow later, where you pick an existing archive as the starting point and replace or add data into that instead of trying to rebuild everything from scratch.

## Features

- Open and browse `.bfz` archives
- Inspect files inside the archive
- View files as hex
- View extracted strings
- Play `.son` audio files
- Export individual files
- Export entire archives to folders
- Export files raw or converted where supported

## Understanding `.bfz` Archives

If you are trying to figure out what a certain `.bfz` archive contains, check the **`WOR Descriptions`** file in the root of the repo.

That file lists known WOR / `.bfz` archive names along with their original paths from the game’s build process.

Those paths came from debug logs found in the game and documented through TCRF. They are useful for making educated guesses about what each archive is for.

## Blender Plugins

The `BlenderPlugins` folder contains Blender add-ons I made for ZOMBI / ZombiU formats.

### `import_zombiu_geo`

`import_zombiu_geo` is a work-in-progress Blender importer for the game’s `.geo` model format.

Current features:

- Imports observed PC `.geo` mesh files
- Imports submeshes as separate Blender objects when possible
- Reads UVs and custom normals
- Uses GEO source names for Blender object names
- Tries to find nearby texture files
- Automatically converts matched `.tdt` textures to PNG when needed
- Assigns found textures as Blender materials

The `.geo` files do not seem to store full texture paths directly, so the importer does its best by looking for related `.tex`, `.tdt`, and converted `.png` files near the model you import.

### Installing the Blender Add-on

1. Open Blender
2. Go to `Edit > Preferences > Add-ons`
3. Click `Install...`
4. Select `import_zombiu_geo.zip` from the `BlenderPlugins` folder
5. Enable `Import-Export: ZombiU / LyN GEO Importer`
6. Use `File > Import > ZombiU / LyN GEO (.geo)`

## Getting Started

### Requirements

- Python 3
- `PySide6`
- `python-lzo`
- `Pillow`
- `numpy`

Some dependencies may only be needed for certain parts of the tool.

### Running the App

1. Download or clone the repository
2. Open a terminal in the `ZOMBIManager` folder
3. Run:

```bash
python zombiManager.py
```

## Project Status

ZOMBI-Manager is still very much a research tool.

Extraction and inspection are the most stable parts. Importing, repacking, and full modding workflows are still unfinished and should be treated as experimental.
