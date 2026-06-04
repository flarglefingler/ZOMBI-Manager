bl_info = {
    "name": "ZombiU Importer",
    "author": "flargle fingler",
    "version": (0, 9, 67),
    "blender": (3, 6, 0),
    "location": "File > Import and View3D Sidebar > ZombiU Importer",
    "description": "Imports ZombiU LyN/Jade worlds, characters, GEO, SKN, TRL, etc.",
    "category": "Import-Export",
}

import importlib
import os
from typing import List

import bpy
from bpy.props import BoolProperty, CollectionProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty
from bpy_extras.io_utils import ImportHelper

# blender keeps modules alive between reloads, so refresh children first.
from . import bfz_archive, blender_import, character_import, game_index, geo_format, material_format, mdf_format, obj_format, resource_index, skn_format, tdt, texture, trl_format, weapon_import, wor_format, world_import

bfz_archive = importlib.reload(bfz_archive)
game_index = importlib.reload(game_index)
geo_format = importlib.reload(geo_format)
material_format = importlib.reload(material_format)
mdf_format = importlib.reload(mdf_format)
obj_format = importlib.reload(obj_format)
resource_index = importlib.reload(resource_index)
skn_format = importlib.reload(skn_format)
trl_format = importlib.reload(trl_format)
wor_format = importlib.reload(wor_format)
tdt = importlib.reload(tdt)
texture = importlib.reload(texture)
blender_import = importlib.reload(blender_import)
world_import = importlib.reload(world_import)
character_import = importlib.reload(character_import)
weapon_import = importlib.reload(weapon_import)

import_geo = blender_import.import_geo
import_skn = blender_import.import_skn
import_trl = blender_import.import_trl
import_trl_debug_variants = blender_import.import_trl_debug_variants
import_world_archive = world_import.import_world_archive
import_character_archive = character_import.import_character_archive
import_weapon = weapon_import.import_weapon
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


def armature_object_poll(_self, obj):
    return obj is not None and obj.type == "ARMATURE"


class IMPORT_OT_zombiu_geo(bpy.types.Operator, ImportHelper):
    bl_idname = "import_scene.zombiu_geo"
    bl_label = "Import ZombiU GEO"
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


class ZOMBI_PG_archive_item(bpy.types.PropertyGroup):
    name: StringProperty()
    path: StringProperty(subtype="FILE_PATH")
    kind: StringProperty()
    file_count: IntProperty()
    world_count: IntProperty()
    geo_count: IntProperty()
    skn_count: IntProperty()
    trl_count: IntProperty()
    tex_count: IntProperty()
    tdt_count: IntProperty()
    mat_count: IntProperty()
    mta_count: IntProperty()
    first_world_name: StringProperty()
    error: StringProperty()


class ZOMBI_PG_character_item(bpy.types.PropertyGroup):
    name: StringProperty()
    path: StringProperty(subtype="FILE_PATH")
    first_world_name: StringProperty()
    skeleton_name: StringProperty()
    geo_count: IntProperty()
    head_count: IntProperty()
    fullbody_count: IntProperty()
    upbody_count: IntProperty()
    lowbody_count: IntProperty()
    arms_count: IntProperty()
    eye_count: IntProperty()
    accessory_count: IntProperty()
    texture_count: IntProperty()
    profile_summary: StringProperty()
    error: StringProperty()


class ZOMBI_PG_weapon_item(bpy.types.PropertyGroup):
    name: StringProperty()
    archive_path: StringProperty(subtype="FILE_PATH")
    archive_name: StringProperty()
    world_name: StringProperty()
    geo_names: StringProperty()
    skn_name: StringProperty()
    geo_count: IntProperty()
    material_count: IntProperty()
    texture_count: IntProperty()
    recommended_bone: StringProperty()
    error: StringProperty()


def _fill_world_scan(scene, summaries) -> int:
    scene.zombiu_archive_items.clear()
    for summary in summaries:
        item = scene.zombiu_archive_items.add()
        item.name = summary.name
        item.path = summary.path
        item.kind = summary.kind
        item.file_count = summary.file_count
        item.world_count = summary.world_count
        item.geo_count = summary.geo_count
        item.skn_count = summary.skn_count
        item.trl_count = summary.trl_count
        item.tex_count = summary.tex_count
        item.tdt_count = summary.tdt_count
        item.mat_count = summary.mat_count
        item.mta_count = summary.mta_count
        item.first_world_name = summary.first_world_name
        item.error = summary.error

    scene.zombiu_archive_index = 0 if scene.zombiu_archive_items else -1
    return sum(1 for item in scene.zombiu_archive_items if item.kind == "World")


def _fill_character_scan(scene, summaries) -> int:
    scene.zombiu_character_items.clear()
    for summary in summaries:
        item = scene.zombiu_character_items.add()
        item.name = summary.name
        item.path = summary.path
        item.first_world_name = summary.first_world_name
        item.skeleton_name = summary.skeleton_name
        item.geo_count = summary.geo_count
        item.head_count = summary.head_count
        item.fullbody_count = summary.fullbody_count
        item.upbody_count = summary.upbody_count
        item.lowbody_count = summary.lowbody_count
        item.arms_count = summary.arms_count
        item.eye_count = summary.eye_count
        item.accessory_count = summary.accessory_count
        item.texture_count = summary.texture_count
        item.profile_summary = summary.profile_summary
        item.error = summary.error

    scene.zombiu_character_index = 0 if scene.zombiu_character_items else -1
    return len(scene.zombiu_character_items)


def _fill_weapon_scan(scene, scan) -> int:
    scene.zombiu_weapon_items.clear()
    if scan is None:
        scene.zombiu_weapon_index = -1
        return 0

    for summary in scan.weapons:
        item = scene.zombiu_weapon_items.add()
        item.name = summary.label
        item.archive_path = summary.archive_path
        item.archive_name = summary.archive_name
        item.world_name = summary.world_name
        item.geo_names = "\n".join(summary.geo_names)
        item.skn_name = summary.skn_name
        item.geo_count = len(summary.geo_names)
        item.material_count = summary.material_count
        item.texture_count = summary.texture_count
        item.recommended_bone = summary.recommended_bone
        item.error = summary.error

    scene.zombiu_weapon_index = 0 if scene.zombiu_weapon_items else -1
    return len(scene.zombiu_weapon_items)


class ZOMBI_UL_archive_list(bpy.types.UIList):
    def filter_items(self, context, data, propname):
        items = getattr(data, propname)
        scene = context.scene
        kind_filter = scene.zombiu_archive_kind_filter
        search = scene.zombiu_archive_search.strip().lower()

        flags = []
        for item in items:
            visible = True
            if kind_filter != "ALL" and item.kind.upper() != kind_filter:
                visible = False
            if search and search not in item.name.lower() and search not in item.first_world_name.lower():
                visible = False
            flags.append(self.bitflag_filter_item if visible else 0)
        return flags, []

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            icon_name = {
                "World": "WORLD_DATA",
                "Common": "FILE",
                "Sound": "SOUND",
                "Video": "FILE_MOVIE",
                "Error": "ERROR",
            }.get(item.kind, "FILE")
            row = layout.row(align=True)
            row.label(text=item.name, icon=icon_name)
            row.label(text=os.path.basename(item.first_world_name) if item.first_world_name else item.kind)
        elif self.layout_type == "GRID":
            layout.alignment = "CENTER"
            layout.label(text="", icon="WORLD_DATA" if item.kind == "World" else "FILE")


class ZOMBI_UL_character_list(bpy.types.UIList):
    def filter_items(self, context, data, propname):
        items = getattr(data, propname)
        search = context.scene.zombiu_character_search.strip().lower()
        flags = []
        for item in items:
            haystack = " ".join((item.name, item.first_world_name, item.skeleton_name, item.profile_summary)).lower()
            visible = not search or search in haystack
            flags.append(self.bitflag_filter_item if visible else 0)
        return flags, []

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            row = layout.row(align=True)
            row.label(text=item.name, icon="ARMATURE_DATA")
            row.label(text=os.path.basename(item.first_world_name) if item.first_world_name else f"{item.geo_count} GEO")
        elif self.layout_type == "GRID":
            layout.alignment = "CENTER"
            layout.label(text="", icon="ARMATURE_DATA")


class ZOMBI_UL_weapon_list(bpy.types.UIList):
    def filter_items(self, context, data, propname):
        items = getattr(data, propname)
        search = context.scene.zombiu_weapon_search.strip().lower()
        flags = []
        for item in items:
            haystack = " ".join((item.name, item.archive_name, item.world_name, item.skn_name)).lower()
            visible = not search or search in haystack
            flags.append(self.bitflag_filter_item if visible else 0)
        return flags, []

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            row = layout.row(align=True)
            row.label(text=item.name, icon="OUTLINER_OB_MESH")
            row.label(text=item.skn_name if item.skn_name else "No SKN")
        elif self.layout_type == "GRID":
            layout.alignment = "CENTER"
            layout.label(text="", icon="OUTLINER_OB_MESH")


class ZOMBI_OT_scan_game_archives(bpy.types.Operator):
    bl_idname = "zombiu.scan_game_archives"
    bl_label = "Scan Worlds"
    bl_options = {"REGISTER"}

    def execute(self, context):
        scene = context.scene
        try:
            summaries = game_index.scan_game_dir(scene.zombiu_game_dir)
        except Exception as exc:
            scene.zombiu_scan_status = str(exc)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        world_count = _fill_world_scan(scene, summaries)
        scene.zombiu_scan_status = f"{len(scene.zombiu_archive_items)} archives, {world_count} worlds"
        self.report({"INFO"}, scene.zombiu_scan_status)
        return {"FINISHED"}


class ZOMBI_OT_scan_characters(bpy.types.Operator):
    bl_idname = "zombiu.scan_characters"
    bl_label = "Scan Characters"
    bl_options = {"REGISTER"}

    def execute(self, context):
        scene = context.scene
        try:
            summaries = character_import.scan_character_archives(scene.zombiu_game_dir)
        except Exception as exc:
            scene.zombiu_character_scan_status = str(exc)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        _fill_character_scan(scene, summaries)
        scene.zombiu_character_scan_status = f"{len(scene.zombiu_character_items)} character archives"
        self.report({"INFO"}, scene.zombiu_character_scan_status)
        return {"FINISHED"}


class ZOMBI_OT_scan_weapons(bpy.types.Operator):
    bl_idname = "zombiu.scan_weapons"
    bl_label = "Scan Weapons"
    bl_options = {"REGISTER"}

    def execute(self, context):
        scene = context.scene
        try:
            scan = weapon_import.scan_weapon_archive(scene.zombiu_game_dir)
        except Exception as exc:
            scene.zombiu_weapon_scan_status = str(exc)
            _fill_weapon_scan(scene, None)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        weapon_count = _fill_weapon_scan(scene, scan)
        scene.zombiu_weapon_scan_status = f"{weapon_count} weapons from {scan.archive_name}"
        self.report({"INFO"}, scene.zombiu_weapon_scan_status)
        return {"FINISHED"}


class ZOMBI_OT_scan_all(bpy.types.Operator):
    bl_idname = "zombiu.scan_all"
    bl_label = "Scan"
    bl_options = {"REGISTER"}

    def execute(self, context):
        scene = context.scene
        try:
            archive_summaries = game_index.scan_game_dir(scene.zombiu_game_dir)
            character_summaries = character_import.scan_character_archives(scene.zombiu_game_dir)
        except Exception as exc:
            message = str(exc)
            scene.zombiu_scan_status = message
            scene.zombiu_character_scan_status = message
            scene.zombiu_weapon_scan_status = message
            self.report({"ERROR"}, message)
            return {"CANCELLED"}

        weapon_scan = None
        weapon_error = ""
        try:
            weapon_scan = weapon_import.scan_weapon_archive(scene.zombiu_game_dir)
        except Exception as exc:
            weapon_error = str(exc)

        world_count = _fill_world_scan(scene, archive_summaries)
        character_count = _fill_character_scan(scene, character_summaries)
        weapon_count = _fill_weapon_scan(scene, weapon_scan)
        scene.zombiu_scan_status = f"{len(scene.zombiu_archive_items)} archives, {world_count} worlds"
        scene.zombiu_character_scan_status = f"{character_count} character archives"
        scene.zombiu_weapon_scan_status = weapon_error or f"{weapon_count} weapons from {weapon_scan.archive_name}"
        self.report({"INFO"}, f"Scanned {world_count} world archive(s), {character_count} character archive(s), {weapon_count} weapon(s)")
        return {"FINISHED"}


class ZOMBI_OT_import_selected_archive(bpy.types.Operator):
    bl_idname = "zombiu.import_selected_archive"
    bl_label = "Import Selected"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        index = scene.zombiu_archive_index
        if index < 0 or index >= len(scene.zombiu_archive_items):
            self.report({"ERROR"}, "Select an archive first")
            return {"CANCELLED"}

        item = scene.zombiu_archive_items[index]
        if item.error:
            self.report({"ERROR"}, item.error)
            return {"CANCELLED"}
        if item.kind != "World":
            self.report({"ERROR"}, "Only world archives can be imported from this panel right now")
            return {"CANCELLED"}

        try:
            roots = import_world_archive(
                scene.zombiu_game_dir,
                item.path,
                include_common=scene.zombiu_mount_common,
                include_sound=scene.zombiu_mount_sound,
                include_video=scene.zombiu_mount_video,
                create_ref_empties=scene.zombiu_world_create_empties,
                empty_limit=scene.zombiu_world_empty_limit,
                import_meshes=scene.zombiu_world_import_meshes,
                object_limit=scene.zombiu_world_object_limit,
                orient_upright=scene.zombiu_world_upright,
                scale=scene.zombiu_world_scale,
                flip_uv_v=True,
                resolve_textures=scene.zombiu_world_resolve_textures,
                convert_tdt_textures=scene.zombiu_world_convert_tdt,
                texture_alpha_mode=scene.zombiu_world_texture_alpha_mode,
            )
        except Exception as exc:
            self.report({"ERROR"}, f"{item.name}: {exc}")
            return {"CANCELLED"}

        resolved = sum(int(root.get("wor_resolved_geo", 0)) for root in roots)
        ambiguous = sum(int(root.get("wor_ambiguous_geo", 0)) for root in roots)
        missing = sum(int(root.get("wor_missing_geo", 0)) for root in roots)
        meshes = sum(int(root.get("wor_imported_geo_instances", 0)) for root in roots)
        self.report({"INFO"}, f"Imported {len(roots)} world(s), {resolved} resolved, {ambiguous} ambiguous, {missing} missing, {meshes} meshes")
        return {"FINISHED"}


class ZOMBI_OT_import_selected_character(bpy.types.Operator):
    bl_idname = "zombiu.import_selected_character"
    bl_label = "Import Character"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        index = scene.zombiu_character_index
        if index < 0 or index >= len(scene.zombiu_character_items):
            self.report({"ERROR"}, "Select a character archive first")
            return {"CANCELLED"}

        item = scene.zombiu_character_items[index]
        if item.error:
            self.report({"ERROR"}, item.error)
            return {"CANCELLED"}

        try:
            objects = import_character_archive(
                scene.zombiu_game_dir,
                item.path,
                include_common=scene.zombiu_mount_common,
                scale=scene.zombiu_character_scale,
                bone_length=scene.zombiu_character_bone_length,
                resolve_textures=scene.zombiu_character_resolve_textures,
                convert_tdt_textures=scene.zombiu_character_convert_tdt,
                texture_alpha_mode=scene.zombiu_character_texture_alpha_mode,
                profile_species=scene.zombiu_character_species,
                profile_sex=scene.zombiu_character_sex,
                profile_body_type=scene.zombiu_character_body_type,
                head_variant=scene.zombiu_character_head_variant,
                body_variant=scene.zombiu_character_body_variant,
                first_person=scene.zombiu_character_first_person,
                part_override=scene.zombiu_character_part_override,
                head_index=scene.zombiu_character_head_index,
                upbody_index=scene.zombiu_character_upbody_index,
                lowbody_index=scene.zombiu_character_lowbody_index,
                arms_index=scene.zombiu_character_arms_index,
                include_eyes=scene.zombiu_character_include_eyes,
                include_accessories=scene.zombiu_character_include_accessories,
                import_all_variants=scene.zombiu_character_all_variants,
            )
        except Exception as exc:
            self.report({"ERROR"}, f"{item.name}: {exc}")
            return {"CANCELLED"}

        self.report({"INFO"}, f"Imported {item.name}: {len(objects)} object(s)")
        return {"FINISHED"}


class ZOMBI_OT_import_selected_weapon(bpy.types.Operator):
    bl_idname = "zombiu.import_selected_weapon"
    bl_label = "Import Weapon"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        index = scene.zombiu_weapon_index
        if index < 0 or index >= len(scene.zombiu_weapon_items):
            self.report({"ERROR"}, "Select a weapon first")
            return {"CANCELLED"}

        item = scene.zombiu_weapon_items[index]
        if item.error:
            self.report({"ERROR"}, item.error)
            return {"CANCELLED"}

        append_armature = None
        if scene.zombiu_weapon_append:
            append_armature = scene.zombiu_weapon_append_armature or active_armature(context) or scene_armature(context)
            if append_armature is None:
                self.report({"ERROR"}, "Pick an append armature or select one in the scene")
                return {"CANCELLED"}

        try:
            objects = import_weapon(
                scene.zombiu_game_dir,
                item.archive_path,
                [name for name in item.geo_names.splitlines() if name],
                skn_name=item.skn_name,
                label=item.name,
                include_common=False,
                scale=scene.zombiu_weapon_scale,
                resolve_textures=scene.zombiu_weapon_resolve_textures,
                convert_tdt_textures=scene.zombiu_weapon_convert_tdt,
                texture_alpha_mode=scene.zombiu_weapon_texture_alpha_mode,
                append_armature=append_armature,
                append_bone=scene.zombiu_weapon_append_bone,
                recommended_bone=item.recommended_bone,
            )
        except Exception as exc:
            self.report({"ERROR"}, f"{item.name}: {exc}")
            return {"CANCELLED"}

        self.report({"INFO"}, f"Imported {item.name}: {len(objects)} object(s)")
        return {"FINISHED"}


class VIEW3D_PT_zombiu_importer(bpy.types.Panel):
    bl_label = "ZombiU Importer"
    bl_idname = "VIEW3D_PT_zombiu_importer"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "ZombiU Importer"

    def draw(self, context):
        scene = context.scene
        layout = self.layout

        layout.prop(scene, "zombiu_game_dir", text="Game Dir")
        layout.operator(ZOMBI_OT_scan_all.bl_idname, icon="FILE_REFRESH")

        box = layout.box()
        box.label(text="Mounted Archives")
        row = box.row(align=True)
        row.prop(scene, "zombiu_mount_common", text="Gen")
        row.prop(scene, "zombiu_mount_sound", text="Snd")
        row.prop(scene, "zombiu_mount_video", text="Vid")


class VIEW3D_PT_zombiu_world_importer(bpy.types.Panel):
    bl_label = "World Importer"
    bl_idname = "VIEW3D_PT_zombiu_world_importer"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "ZombiU Importer"
    bl_parent_id = "VIEW3D_PT_zombiu_importer"

    def draw(self, context):
        scene = context.scene
        layout = self.layout

        row = layout.row(align=True)
        row.prop(scene, "zombiu_archive_kind_filter", text="")
        row.prop(scene, "zombiu_archive_search", text="", icon="VIEWZOOM")
        if scene.zombiu_scan_status:
            layout.label(text=scene.zombiu_scan_status)

        layout.template_list(
            "ZOMBI_UL_archive_list",
            "",
            scene,
            "zombiu_archive_items",
            scene,
            "zombiu_archive_index",
            rows=8,
        )

        selected = None
        if 0 <= scene.zombiu_archive_index < len(scene.zombiu_archive_items):
            selected = scene.zombiu_archive_items[scene.zombiu_archive_index]

        if selected:
            box = layout.box()
            box.label(text="Selected World", icon="WORLD_DATA")
            box.label(text=f"Archive: {selected.name}")
            if selected.first_world_name:
                box.label(text=f"World: {os.path.basename(selected.first_world_name)}")
            box.label(text=f"Files: {selected.file_count}")
            if selected.error:
                box.label(text=selected.error, icon="ERROR")
            else:
                box.label(text=f"Geometry Files: {selected.geo_count}")
                box.label(text=f"MAT Files: {selected.mat_count}   MTA Files: {selected.mta_count}")
                box.label(text=f"TEX Files: {selected.tex_count}   TDT Files: {selected.tdt_count}")

        import_box = layout.box()
        import_box.label(text="Import Options")
        import_box.prop(scene, "zombiu_world_scale", text="Scale")
        import_box.prop(scene, "zombiu_world_object_limit", text="Mesh Cap")
        row = import_box.row(align=True)
        row.prop(scene, "zombiu_world_import_meshes", text="Meshes")
        row.prop(scene, "zombiu_world_upright", text="Upright")

        material_box = layout.box()
        material_box.label(text="Materials")
        row = material_box.row(align=True)
        row.prop(scene, "zombiu_world_resolve_textures", text="Resolve")
        row.prop(scene, "zombiu_world_convert_tdt", text="TDT")
        material_box.prop(scene, "zombiu_world_texture_alpha_mode", text="Alpha")

        debug_box = layout.box()
        debug_box.label(text="Debug")
        debug_box.prop(scene, "zombiu_world_create_empties", text="Reference Empties")
        if scene.zombiu_world_create_empties:
            debug_box.prop(scene, "zombiu_world_empty_limit", text="Empty Cap")

        layout.operator(ZOMBI_OT_import_selected_archive.bl_idname, icon="IMPORT")


class VIEW3D_PT_zombiu_character_importer(bpy.types.Panel):
    bl_label = "Character Importer"
    bl_idname = "VIEW3D_PT_zombiu_character_importer"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "ZombiU Importer"
    bl_parent_id = "VIEW3D_PT_zombiu_importer"

    def draw(self, context):
        scene = context.scene
        layout = self.layout

        row = layout.row(align=True)
        row.prop(scene, "zombiu_character_search", text="", icon="VIEWZOOM")
        if scene.zombiu_character_scan_status:
            layout.label(text=scene.zombiu_character_scan_status)

        layout.template_list(
            "ZOMBI_UL_character_list",
            "",
            scene,
            "zombiu_character_items",
            scene,
            "zombiu_character_index",
            rows=7,
        )

        selected = None
        if 0 <= scene.zombiu_character_index < len(scene.zombiu_character_items):
            selected = scene.zombiu_character_items[scene.zombiu_character_index]

        if selected:
            box = layout.box()
            box.label(text="Selected Character Archive", icon="ARMATURE_DATA")
            box.label(text=f"Archive: {selected.name}")
            if selected.first_world_name:
                box.label(text=f"World: {os.path.basename(selected.first_world_name)}")
            box.label(text=f"Skeleton: {selected.skeleton_name}")
            box.label(text=f"Character GEO Files: {selected.geo_count}")
            box.label(text=f"Heads: {selected.head_count}")
            box.label(text=f"Full Bodies: {selected.fullbody_count}")
            box.label(text=f"Upper Bodies: {selected.upbody_count}")
            box.label(text=f"Lower Bodies: {selected.lowbody_count}")
            box.label(text=f"First-Person Arms: {selected.arms_count}")
            box.label(text=f"Eyes: {selected.eye_count}   Accessories: {selected.accessory_count}")
            if selected.profile_summary:
                box.label(text=f"Profiles: {selected.profile_summary}")
            box.label(text=f"Texture Files: {selected.texture_count}")

        mode_box = layout.box()
        mode_box.label(text="Import Mode")
        mode_box.prop(scene, "zombiu_character_first_person", text="First Person")
        if scene.zombiu_character_first_person:
            mode_box.label(text="Human arms only")

        filter_box = layout.box()
        filter_box.label(text="Character Filter")
        type_col = filter_box.column(align=True)
        type_col.enabled = not scene.zombiu_character_first_person
        type_col.prop(scene, "zombiu_character_species", text="Type")
        filter_box.prop(scene, "zombiu_character_sex", text="Sex")
        filter_box.prop(scene, "zombiu_character_body_type", text="Body")

        variant_box = layout.box()
        variant_box.label(text="Variants")
        variant_box.prop(scene, "zombiu_character_body_variant", text="Outfit Variant")
        head_row = variant_box.row()
        head_row.enabled = not scene.zombiu_character_first_person
        head_row.prop(scene, "zombiu_character_head_variant", text="Head Variant")

        parts_box = layout.box()
        parts_box.label(text="Included Parts")
        row = parts_box.row(align=True)
        row.enabled = not scene.zombiu_character_first_person
        row.prop(scene, "zombiu_character_include_eyes", text="Eyes")
        row.prop(scene, "zombiu_character_include_accessories", text="Accessories")
        row = parts_box.row(align=True)
        row.prop(scene, "zombiu_character_resolve_textures", text="Materials")
        row.prop(scene, "zombiu_character_convert_tdt", text="Convert TDT")

        advanced_box = layout.box()
        advanced_box.prop(scene, "zombiu_character_part_override", text="Advanced Part Slots")
        if scene.zombiu_character_part_override:
            row = advanced_box.row(align=True)
            if scene.zombiu_character_first_person:
                row.prop(scene, "zombiu_character_arms_index", text="Arms")
            else:
                row.prop(scene, "zombiu_character_head_index", text="Head")
                row.prop(scene, "zombiu_character_upbody_index", text="Up")
                row = advanced_box.row(align=True)
                row.prop(scene, "zombiu_character_lowbody_index", text="Low")
                row.prop(scene, "zombiu_character_arms_index", text="Arms")
        advanced_box.prop(scene, "zombiu_character_scale", text="Scale")
        advanced_box.prop(scene, "zombiu_character_bone_length", text="Bone")
        advanced_box.prop(scene, "zombiu_character_texture_alpha_mode", text="Alpha")
        advanced_box.prop(scene, "zombiu_character_all_variants", text="Import All Matching Parts")

        layout.operator(ZOMBI_OT_import_selected_character.bl_idname, icon="IMPORT")


class VIEW3D_PT_zombiu_weapon_importer(bpy.types.Panel):
    bl_label = "Weapons"
    bl_idname = "VIEW3D_PT_zombiu_weapon_importer"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "ZombiU Importer"
    bl_parent_id = "VIEW3D_PT_zombiu_importer"

    def draw(self, context):
        scene = context.scene
        layout = self.layout

        row = layout.row(align=True)
        row.prop(scene, "zombiu_weapon_search", text="", icon="VIEWZOOM")
        if scene.zombiu_weapon_scan_status:
            layout.label(text=scene.zombiu_weapon_scan_status)

        layout.template_list(
            "ZOMBI_UL_weapon_list",
            "",
            scene,
            "zombiu_weapon_items",
            scene,
            "zombiu_weapon_index",
            rows=7,
        )

        selected = None
        if 0 <= scene.zombiu_weapon_index < len(scene.zombiu_weapon_items):
            selected = scene.zombiu_weapon_items[scene.zombiu_weapon_index]

        if selected:
            box = layout.box()
            box.label(text="Selected Weapon", icon="OUTLINER_OB_MESH")
            box.label(text=f"Weapon: {selected.name}")
            box.label(text=f"Archive: {selected.archive_name}")
            if selected.world_name:
                box.label(text=f"World: {os.path.basename(selected.world_name)}")
            box.label(text=f"Geometry Files: {selected.geo_count}")
            box.label(text=f"Skeleton: {selected.skn_name if selected.skn_name else 'None'}")
            box.label(text=f"Suggested Bone: {selected.recommended_bone}")
            box.label(text=f"MAT/MTA Files: {selected.material_count}   Texture Files: {selected.texture_count}")

        import_box = layout.box()
        import_box.label(text="Import Options")
        import_box.prop(scene, "zombiu_weapon_scale", text="Scale")
        row = import_box.row(align=True)
        row.prop(scene, "zombiu_weapon_resolve_textures", text="Materials")
        row.prop(scene, "zombiu_weapon_convert_tdt", text="Convert TDT")
        import_box.prop(scene, "zombiu_weapon_texture_alpha_mode", text="Alpha")

        append_box = layout.box()
        append_box.label(text="Append")

        # TODO: Attaching to characters isn't correct so some animations are ass
        append_box.prop(scene, "zombiu_weapon_append", text="Attach To Character")
        if scene.zombiu_weapon_append:
            append_box.prop(scene, "zombiu_weapon_append_armature", text="Rig")
            if scene.zombiu_weapon_append_armature:
                append_box.prop_search(
                    scene,
                    "zombiu_weapon_append_bone",
                    scene.zombiu_weapon_append_armature.data,
                    "bones",
                    text="Append Bone",
                )
            else:
                append_box.prop(scene, "zombiu_weapon_append_bone", text="Append Bone")
            if selected and scene.zombiu_weapon_append_bone.strip().lower() == "auto":
                append_box.label(text=f"Auto uses {selected.recommended_bone}")

        layout.operator(ZOMBI_OT_import_selected_weapon.bl_idname, icon="IMPORT")


def menu_func_import_geo(self, context):
    self.layout.operator(IMPORT_OT_zombiu_geo.bl_idname, text="ZombiU Importer GEO (.geo)")


def menu_func_import_skn(self, context):
    self.layout.operator(IMPORT_OT_zombiu_skn.bl_idname, text="ZombiU Importer SKN (.skn)")


def menu_func_import_trl(self, context):
    self.layout.operator(IMPORT_OT_zombiu_trl.bl_idname, text="ZombiU Importer TRL (.trl)")


def register():
    bpy.utils.register_class(IMPORT_OT_zombiu_geo)
    bpy.utils.register_class(IMPORT_OT_zombiu_skn)
    bpy.utils.register_class(IMPORT_OT_zombiu_trl)
    bpy.utils.register_class(ZOMBI_PG_archive_item)
    bpy.utils.register_class(ZOMBI_PG_character_item)
    bpy.utils.register_class(ZOMBI_PG_weapon_item)
    bpy.utils.register_class(ZOMBI_UL_archive_list)
    bpy.utils.register_class(ZOMBI_UL_character_list)
    bpy.utils.register_class(ZOMBI_UL_weapon_list)
    bpy.utils.register_class(ZOMBI_OT_scan_game_archives)
    bpy.utils.register_class(ZOMBI_OT_scan_characters)
    bpy.utils.register_class(ZOMBI_OT_scan_weapons)
    bpy.utils.register_class(ZOMBI_OT_scan_all)
    bpy.utils.register_class(ZOMBI_OT_import_selected_archive)
    bpy.utils.register_class(ZOMBI_OT_import_selected_character)
    bpy.utils.register_class(ZOMBI_OT_import_selected_weapon)
    bpy.utils.register_class(VIEW3D_PT_zombiu_importer)
    bpy.utils.register_class(VIEW3D_PT_zombiu_world_importer)
    bpy.utils.register_class(VIEW3D_PT_zombiu_character_importer)
    bpy.utils.register_class(VIEW3D_PT_zombiu_weapon_importer)

    bpy.types.Scene.zombiu_game_dir = StringProperty(name="Game Dir", subtype="DIR_PATH")
    bpy.types.Scene.zombiu_archive_items = CollectionProperty(type=ZOMBI_PG_archive_item)
    bpy.types.Scene.zombiu_archive_index = IntProperty(default=-1)
    bpy.types.Scene.zombiu_archive_search = StringProperty(name="Search")
    bpy.types.Scene.zombiu_scan_status = StringProperty(name="Status")
    bpy.types.Scene.zombiu_archive_kind_filter = EnumProperty(
        name="Filter",
        items=(
            ("ALL", "All", ""),
            ("WORLD", "Worlds", ""),
            ("COMMON", "Common", ""),
            ("ARCHIVE", "Other", ""),
            ("SOUND", "Sound", ""),
            ("VIDEO", "Video", ""),
            ("ERROR", "Errors", ""),
        ),
        default="WORLD",
    )
    bpy.types.Scene.zombiu_mount_common = BoolProperty(name="Gen Common", default=True)
    bpy.types.Scene.zombiu_mount_sound = BoolProperty(name="SndStream", default=False)
    bpy.types.Scene.zombiu_mount_video = BoolProperty(name="Video", default=False)
    bpy.types.Scene.zombiu_world_scale = FloatProperty(
        name="Scale",
        default=1.0,
        min=0.0001,
        soft_min=0.01,
        soft_max=100.0,
    )
    bpy.types.Scene.zombiu_world_object_limit = IntProperty(
        name="Limit",
        default=0,
        min=0,
        soft_max=5000,
    )
    bpy.types.Scene.zombiu_world_import_meshes = BoolProperty(name="Meshes", default=True)
    bpy.types.Scene.zombiu_world_upright = BoolProperty(name="Upright", default=True)
    bpy.types.Scene.zombiu_world_resolve_textures = BoolProperty(name="Materials", default=True)
    bpy.types.Scene.zombiu_world_convert_tdt = BoolProperty(name="TDT", default=True)
    bpy.types.Scene.zombiu_world_texture_alpha_mode = EnumProperty(
        name="Alpha",
        items=(
            ("opaque", "Opaque", ""),
            ("clip", "Clip", ""),
            ("blend", "Blend", ""),
        ),
        default="opaque",
    )
    bpy.types.Scene.zombiu_world_create_empties = BoolProperty(name="Empties", default=False)
    bpy.types.Scene.zombiu_world_empty_limit = IntProperty(name="Empty Cap", default=250, min=0, soft_max=5000)

    bpy.types.Scene.zombiu_character_items = CollectionProperty(type=ZOMBI_PG_character_item)
    bpy.types.Scene.zombiu_character_index = IntProperty(default=-1)
    bpy.types.Scene.zombiu_character_search = StringProperty(name="Search")
    bpy.types.Scene.zombiu_character_scan_status = StringProperty(name="Status")
    bpy.types.Scene.zombiu_character_scale = FloatProperty(
        name="Scale",
        default=1.0,
        min=0.0001,
        soft_min=0.01,
        soft_max=100.0,
    )
    bpy.types.Scene.zombiu_character_bone_length = FloatProperty(
        name="Bone",
        default=0.05,
        min=0.001,
        soft_min=0.01,
        soft_max=1.0,
    )
    bpy.types.Scene.zombiu_character_species = EnumProperty(
        name="Type",
        items=(
            ("auto", "Auto", ""),
            ("human", "Human", ""),
            ("zombie", "Zombie", ""),
        ),
        default="auto",
    )
    bpy.types.Scene.zombiu_character_sex = EnumProperty(
        name="Sex",
        items=(
            ("auto", "Auto", ""),
            ("male", "Male", ""),
            ("female", "Female", ""),
        ),
        default="auto",
    )
    bpy.types.Scene.zombiu_character_body_type = EnumProperty(
        name="Body",
        items=(
            ("auto", "Auto", ""),
            ("regular", "Regular", ""),
            ("fat", "Fat", ""),
            ("thin", "Thin", ""),
        ),
        default="auto",
    )
    bpy.types.Scene.zombiu_character_head_variant = IntProperty(name="Head Variant", default=0, min=0, soft_max=32)
    bpy.types.Scene.zombiu_character_body_variant = IntProperty(name="Outfit Variant", default=0, min=0, soft_max=32)
    bpy.types.Scene.zombiu_character_first_person = BoolProperty(
        name="First Person",
        description="Import the matching human first-person arms only",
        default=False,
    )
    bpy.types.Scene.zombiu_character_part_override = BoolProperty(name="Advanced Part Slots", default=False)
    bpy.types.Scene.zombiu_character_head_index = IntProperty(name="Head", default=0, min=0, soft_max=32)
    bpy.types.Scene.zombiu_character_upbody_index = IntProperty(name="Up", default=0, min=0, soft_max=32)
    bpy.types.Scene.zombiu_character_lowbody_index = IntProperty(name="Low", default=0, min=0, soft_max=32)
    bpy.types.Scene.zombiu_character_arms_index = IntProperty(name="Arms", default=0, min=0, soft_max=32)
    bpy.types.Scene.zombiu_character_include_eyes = BoolProperty(name="Eyes", default=True)
    bpy.types.Scene.zombiu_character_include_accessories = BoolProperty(name="Accessories", default=True)
    bpy.types.Scene.zombiu_character_all_variants = BoolProperty(name="All Variants", default=False)
    bpy.types.Scene.zombiu_character_resolve_textures = BoolProperty(name="Materials", default=True)
    bpy.types.Scene.zombiu_character_convert_tdt = BoolProperty(name="TDT", default=True)
    bpy.types.Scene.zombiu_character_texture_alpha_mode = EnumProperty(
        name="Alpha",
        items=(
            ("opaque", "Opaque", ""),
            ("clip", "Clip", ""),
            ("blend", "Blend", ""),
        ),
        default="opaque",
    )
    bpy.types.Scene.zombiu_weapon_items = CollectionProperty(type=ZOMBI_PG_weapon_item)
    bpy.types.Scene.zombiu_weapon_index = IntProperty(default=-1)
    bpy.types.Scene.zombiu_weapon_search = StringProperty(name="Search")
    bpy.types.Scene.zombiu_weapon_scan_status = StringProperty(name="Status")
    bpy.types.Scene.zombiu_weapon_scale = FloatProperty(
        name="Scale",
        default=1.0,
        min=0.0001,
        soft_min=0.01,
        soft_max=100.0,
    )
    bpy.types.Scene.zombiu_weapon_resolve_textures = BoolProperty(name="Materials", default=True)
    bpy.types.Scene.zombiu_weapon_convert_tdt = BoolProperty(name="TDT", default=True)
    bpy.types.Scene.zombiu_weapon_texture_alpha_mode = EnumProperty(
        name="Alpha",
        items=(
            ("opaque", "Opaque", ""),
            ("clip", "Clip", ""),
            ("blend", "Blend", ""),
        ),
        default="opaque",
    )
    bpy.types.Scene.zombiu_weapon_append = BoolProperty(name="Append", default=False)
    bpy.types.Scene.zombiu_weapon_append_armature = PointerProperty(
        name="Append Rig",
        type=bpy.types.Object,
        poll=armature_object_poll,
    )
    bpy.types.Scene.zombiu_weapon_append_bone = StringProperty(name="Append Bone", default="Auto")
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import_geo)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import_skn)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import_trl)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import_trl)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import_skn)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import_geo)

    del bpy.types.Scene.zombiu_world_empty_limit
    del bpy.types.Scene.zombiu_world_create_empties
    del bpy.types.Scene.zombiu_world_texture_alpha_mode
    del bpy.types.Scene.zombiu_world_convert_tdt
    del bpy.types.Scene.zombiu_world_resolve_textures
    del bpy.types.Scene.zombiu_world_upright
    del bpy.types.Scene.zombiu_world_import_meshes
    del bpy.types.Scene.zombiu_world_object_limit
    del bpy.types.Scene.zombiu_world_scale
    del bpy.types.Scene.zombiu_weapon_append_bone
    del bpy.types.Scene.zombiu_weapon_append_armature
    del bpy.types.Scene.zombiu_weapon_append
    del bpy.types.Scene.zombiu_weapon_texture_alpha_mode
    del bpy.types.Scene.zombiu_weapon_convert_tdt
    del bpy.types.Scene.zombiu_weapon_resolve_textures
    del bpy.types.Scene.zombiu_weapon_scale
    del bpy.types.Scene.zombiu_weapon_scan_status
    del bpy.types.Scene.zombiu_weapon_search
    del bpy.types.Scene.zombiu_weapon_index
    del bpy.types.Scene.zombiu_weapon_items
    del bpy.types.Scene.zombiu_character_texture_alpha_mode
    del bpy.types.Scene.zombiu_character_convert_tdt
    del bpy.types.Scene.zombiu_character_resolve_textures
    del bpy.types.Scene.zombiu_character_all_variants
    del bpy.types.Scene.zombiu_character_include_accessories
    del bpy.types.Scene.zombiu_character_include_eyes
    del bpy.types.Scene.zombiu_character_part_override
    del bpy.types.Scene.zombiu_character_first_person
    del bpy.types.Scene.zombiu_character_body_variant
    del bpy.types.Scene.zombiu_character_head_variant
    del bpy.types.Scene.zombiu_character_body_type
    del bpy.types.Scene.zombiu_character_sex
    del bpy.types.Scene.zombiu_character_species
    del bpy.types.Scene.zombiu_character_arms_index
    del bpy.types.Scene.zombiu_character_lowbody_index
    del bpy.types.Scene.zombiu_character_upbody_index
    del bpy.types.Scene.zombiu_character_head_index
    del bpy.types.Scene.zombiu_character_bone_length
    del bpy.types.Scene.zombiu_character_scale
    del bpy.types.Scene.zombiu_character_scan_status
    del bpy.types.Scene.zombiu_character_search
    del bpy.types.Scene.zombiu_character_index
    del bpy.types.Scene.zombiu_character_items
    del bpy.types.Scene.zombiu_mount_video
    del bpy.types.Scene.zombiu_mount_sound
    del bpy.types.Scene.zombiu_mount_common
    del bpy.types.Scene.zombiu_archive_kind_filter
    del bpy.types.Scene.zombiu_scan_status
    del bpy.types.Scene.zombiu_archive_search
    del bpy.types.Scene.zombiu_archive_index
    del bpy.types.Scene.zombiu_archive_items
    del bpy.types.Scene.zombiu_game_dir

    bpy.utils.unregister_class(VIEW3D_PT_zombiu_weapon_importer)
    bpy.utils.unregister_class(VIEW3D_PT_zombiu_character_importer)
    bpy.utils.unregister_class(VIEW3D_PT_zombiu_world_importer)
    bpy.utils.unregister_class(VIEW3D_PT_zombiu_importer)
    bpy.utils.unregister_class(ZOMBI_OT_import_selected_weapon)
    bpy.utils.unregister_class(ZOMBI_OT_import_selected_character)
    bpy.utils.unregister_class(ZOMBI_OT_import_selected_archive)
    bpy.utils.unregister_class(ZOMBI_OT_scan_all)
    bpy.utils.unregister_class(ZOMBI_OT_scan_weapons)
    bpy.utils.unregister_class(ZOMBI_OT_scan_characters)
    bpy.utils.unregister_class(ZOMBI_OT_scan_game_archives)
    bpy.utils.unregister_class(ZOMBI_UL_weapon_list)
    bpy.utils.unregister_class(ZOMBI_UL_character_list)
    bpy.utils.unregister_class(ZOMBI_UL_archive_list)
    bpy.utils.unregister_class(ZOMBI_PG_weapon_item)
    bpy.utils.unregister_class(ZOMBI_PG_character_item)
    bpy.utils.unregister_class(ZOMBI_PG_archive_item)
    bpy.utils.unregister_class(IMPORT_OT_zombiu_trl)
    bpy.utils.unregister_class(IMPORT_OT_zombiu_skn)
    bpy.utils.unregister_class(IMPORT_OT_zombiu_geo)


if __name__ == "__main__":
    register()
