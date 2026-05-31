bl_info = {
    "name": "ZombiU / LyN GEO Importer",
    "author": "flargle fingler",
    "version": (0, 8, 1),
    "blender": (3, 6, 0),
    "location": "File > Import > ZombiU / LyN GEO (.geo)",
    "description": "Imports ZombiU LyN/Jade .geo VISU mesh files. (Note: Only tested on PC files, use at ur own risk)",
    "category": "Import-Export",
}

import importlib
import os
from typing import List

import bpy
from bpy.props import BoolProperty, CollectionProperty, FloatProperty, StringProperty
from bpy_extras.io_utils import ImportHelper

# blender keeps modules alive between reloads, so refresh children first.
if "geo_format" in locals():
    importlib.reload(geo_format)
    importlib.reload(tdt)
    importlib.reload(texture)
    importlib.reload(blender_import)
else:
    from . import geo_format, tdt, texture, blender_import

import_geo = blender_import.import_geo

class IMPORT_OT_zombiu_geo(bpy.types.Operator, ImportHelper):
    bl_idname = "import_scene.zombiu_geo"
    bl_label = "Import ZombiU / LyN GEO"
    bl_options = {"PRESET", "UNDO"}

    filename_ext = ".geo"
    filter_glob: StringProperty(default="*.geo", options={"HIDDEN"})
    files: CollectionProperty(type=bpy.types.OperatorFileListElement, options={"HIDDEN", "SKIP_SAVE"})
    directory: StringProperty(subtype="DIR_PATH", options={"HIDDEN"})

    scale: FloatProperty(
        name="Scale",
        description="Scale applied to imported positions",
        default=1.0,
        min=0.0001,
        soft_min=0.01,
        soft_max=100.0,
    )

    flip_uv_v: BoolProperty(
        name="Flip UV V",
        description="Flip texture V coordinates for Blender's UV convention",
        default=True,
    )

    resolve_textures: BoolProperty(
        name="Auto Resolve Textures",
        description="Search nearby TEX/TDT/PNG files and assign material when matched",
        default=True,
    )

    convert_tdt_textures: BoolProperty(
        name="Convert TDT Textures",
        description="Convert matched .PC.tdt file to PNG in a converted_textures folder when no PNG already exists",
        default=True,
    )

    def execute(self, context):
        filepaths = []
        if len(self.files) > 0:
            for file_entry in self.files:
                name = file_entry.name
                filepaths.append(name if os.path.isabs(name) else os.path.join(self.directory, name))
        else:
            filepaths.append(self.filepath)

        objects: List[bpy.types.Object] = []
        errors = []
        for filepath in filepaths:
            if not filepath.lower().endswith(".geo"):
                continue
            try:
                objects.extend(
                    import_geo(
                        filepath,
                        self.scale,
                        self.flip_uv_v,
                        self.resolve_textures,
                        self.convert_tdt_textures,
                    )
                )
            except Exception as exc:
                errors.append(f"{os.path.basename(filepath)}: {exc}")

        if errors and not objects:
            self.report({"ERROR"}, "; ".join(errors[:3]))
            return {"CANCELLED"}

        if errors:
            self.report({"WARNING"}, f"Imported with {len(errors)} error(s): {'; '.join(errors[:2])}")

        face_count = sum(len(obj.data.polygons) for obj in objects)
        self.report({"INFO"}, f"Imported {len(filepaths)} GEO file(s), {len(objects)} part(s), {face_count} faces")
        return {"FINISHED"}


def menu_func_import(self, context):
    self.layout.operator(IMPORT_OT_zombiu_geo.bl_idname, text="ZombiU / LyN GEO (.geo)")


def register():
    bpy.utils.register_class(IMPORT_OT_zombiu_geo)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    bpy.utils.unregister_class(IMPORT_OT_zombiu_geo)


if __name__ == "__main__":
    register()
