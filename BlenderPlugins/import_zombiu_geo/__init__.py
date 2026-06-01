bl_info = {
    "name": "ZombiU / LyN Importers",
    "author": "flargle fingler",
    "version": (0, 9, 32),
    "blender": (3, 6, 0),
    "location": "File > Import > ZombiU / LyN",
    "description": "Imports observed ZombiU LyN/Jade GEO, SKN, and TRL files. PC files only so far, use at ur own risk",
    "category": "Import-Export",
}

import importlib
import os
from typing import List

import bpy
from bpy.props import BoolProperty, CollectionProperty, EnumProperty, FloatProperty, StringProperty
from bpy_extras.io_utils import ImportHelper

# blender keeps modules alive between reloads, so refresh children first.
from . import blender_import, geo_format, skn_format, tdt, texture, trl_format

geo_format = importlib.reload(geo_format)
skn_format = importlib.reload(skn_format)
trl_format = importlib.reload(trl_format)
tdt = importlib.reload(tdt)
texture = importlib.reload(texture)
blender_import = importlib.reload(blender_import)

import_geo = blender_import.import_geo
import_skn = blender_import.import_skn
import_trl = blender_import.import_trl
import_trl_debug_variants = blender_import.import_trl_debug_variants
active_armature = blender_import.active_armature
scene_armature = blender_import.scene_armature


def selected_filepaths(operator) -> List[str]:
    if len(operator.files) > 0:
        paths = []
        for file_entry in operator.files:
            name = file_entry.name
            paths.append(name if os.path.isabs(name) else os.path.join(operator.directory, name))
        return paths
    return [operator.filepath]


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

    texture_alpha_mode: EnumProperty(
        name="Texture Alpha",
        description="How to use alpha from matched DA/ENCA textures",
        items=(
            ("opaque", "Opaque", "Ignore texture alpha. Best for most character clothing/body textures"),
            ("clip", "Alpha Clip", "Use alpha as a hard cutout"),
            ("blend", "Alpha Blend", "Use alpha as real transparency"),
        ),
        default="opaque",
    )

    armature_link: EnumProperty(
        name="Rig",
        description="Parent imported parts to an armature",
        items=(
            ("none", "None", "Do not link to an armature"),
            ("selected", "Selected", "Use the selected armature"),
            ("scene", "Only Rig", "Use the only armature in the scene"),
        ),
        default="none",
    )

    add_armature_modifier: BoolProperty(
        name="Modifier",
        description="Add an Armature modifier while linking",
        default=True,
    )

    def execute(self, context):
        filepaths = selected_filepaths(self)
        armature_object = None
        if self.armature_link == "selected":
            armature_object = active_armature(context)
            if armature_object is None:
                self.report({"WARNING"}, "No selected armature; importing unlinked")
        elif self.armature_link == "scene":
            armature_object = scene_armature(context)
            if armature_object is None:
                self.report({"WARNING"}, "Scene needs exactly one armature; importing unlinked")

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
                        self.texture_alpha_mode,
                        armature_object,
                        self.add_armature_modifier,
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
        unweighted = sum(1 for obj in objects if obj.get("geo_armature_warning"))
        if unweighted:
            self.report(
                {"WARNING"},
                f"Imported {len(filepaths)} GEO file(s), {len(objects)} part(s), {face_count} faces; "
                f"{unweighted} linked part(s) had no matching skin weights",
            )
        else:
            self.report({"INFO"}, f"Imported {len(filepaths)} GEO file(s), {len(objects)} part(s), {face_count} faces")
        return {"FINISHED"}


class IMPORT_OT_zombiu_skn(bpy.types.Operator, ImportHelper):
    bl_idname = "import_scene.zombiu_skn"
    bl_label = "Import ZombiU SKN"
    bl_options = {"PRESET", "UNDO"}

    filename_ext = ".skn"
    filter_glob: StringProperty(default="*.skn", options={"HIDDEN"})
    files: CollectionProperty(type=bpy.types.OperatorFileListElement, options={"HIDDEN", "SKIP_SAVE"})
    directory: StringProperty(subtype="DIR_PATH", options={"HIDDEN"})

    scale: FloatProperty(
        name="Scale",
        description="Scale applied to imported bone positions",
        default=1.0,
        min=0.0001,
        soft_min=0.01,
        soft_max=100.0,
    )

    bone_length: FloatProperty(
        name="Leaf Size",
        description="Display length for bones without children",
        default=0.05,
        min=0.001,
        soft_max=1.0,
    )

    def execute(self, context):
        filepaths = selected_filepaths(self)
        objects: List[bpy.types.Object] = []
        errors = []

        for filepath in filepaths:
            if not filepath.lower().endswith(".skn"):
                continue
            try:
                objects.append(import_skn(filepath, self.scale, self.bone_length))
            except Exception as exc:
                errors.append(f"{os.path.basename(filepath)}: {exc}")

        if errors and not objects:
            self.report({"ERROR"}, "; ".join(errors[:3]))
            return {"CANCELLED"}
        if errors:
            self.report({"WARNING"}, f"Imported with {len(errors)} error(s): {'; '.join(errors[:2])}")

        self.report({"INFO"}, f"Imported {len(objects)} SKN armature(s)")
        return {"FINISHED"}


class IMPORT_OT_zombiu_trl(bpy.types.Operator, ImportHelper):
    bl_idname = "import_scene.zombiu_trl"
    bl_label = "Import ZombiU TRL"
    bl_options = {"PRESET", "UNDO"}

    filename_ext = ".trl"
    filter_glob: StringProperty(default="*.trl;*.PC.trl", options={"HIDDEN"})
    files: CollectionProperty(type=bpy.types.OperatorFileListElement, options={"HIDDEN", "SKIP_SAVE"})
    directory: StringProperty(subtype="DIR_PATH", options={"HIDDEN"})

    assign_action: BoolProperty(
        name="Assign",
        description="Set the imported action on the selected armature",
        default=True,
    )

    set_scene_fps: BoolProperty(
        name="Set FPS",
        description="Set the scene frame range and FPS from the TRL",
        default=False,
    )

    bake_sampled_frames: BoolProperty(
        name="Fallback Bake",
        description="If packed keys cannot be decoded, fill the gaps by interpolating coarse sampled poses",
        default=False,
    )

    debug_rotations: BoolProperty(
        name="Debug Rot",
        description="Import several rotation decode candidates as separate actions",
        default=False,
    )

    def execute(self, context):
        armature_object = active_armature(context)
        if armature_object is None:
            self.report({"ERROR"}, "Select an imported SKN armature first")
            return {"CANCELLED"}

        filepaths = selected_filepaths(self)
        actions: List[bpy.types.Action] = []
        errors = []

        for filepath in filepaths:
            if not filepath.lower().endswith(".trl"):
                continue
            try:
                if self.debug_rotations:
                    actions.extend(
                        import_trl_debug_variants(
                            filepath,
                            armature_object,
                            self.assign_action,
                            self.set_scene_fps,
                            self.bake_sampled_frames,
                        )
                    )
                else:
                    actions.append(
                        import_trl(
                            filepath,
                            armature_object,
                            self.assign_action,
                            self.set_scene_fps,
                            bake_sampled_frames=self.bake_sampled_frames,
                        )
                    )
            except Exception as exc:
                errors.append(f"{os.path.basename(filepath)}: {exc}")

        if errors and not actions:
            self.report({"ERROR"}, "; ".join(errors[:3]))
            return {"CANCELLED"}
        if errors:
            self.report({"WARNING"}, f"Imported with {len(errors)} error(s): {'; '.join(errors[:2])}")

        self.report({"INFO"}, f"Imported {len(actions)} TRL action(s)")
        return {"FINISHED"}


def menu_func_import_geo(self, context):
    self.layout.operator(IMPORT_OT_zombiu_geo.bl_idname, text="ZombiU / LyN GEO (.geo)")


def menu_func_import_skn(self, context):
    self.layout.operator(IMPORT_OT_zombiu_skn.bl_idname, text="ZombiU / LyN SKN (.skn)")


def menu_func_import_trl(self, context):
    self.layout.operator(IMPORT_OT_zombiu_trl.bl_idname, text="ZombiU / LyN TRL (.trl)")


def register():
    bpy.utils.register_class(IMPORT_OT_zombiu_geo)
    bpy.utils.register_class(IMPORT_OT_zombiu_skn)
    bpy.utils.register_class(IMPORT_OT_zombiu_trl)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import_geo)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import_skn)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import_trl)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import_trl)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import_skn)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import_geo)
    bpy.utils.unregister_class(IMPORT_OT_zombiu_trl)
    bpy.utils.unregister_class(IMPORT_OT_zombiu_skn)
    bpy.utils.unregister_class(IMPORT_OT_zombiu_geo)


if __name__ == "__main__":
    register()
