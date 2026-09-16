"""Generate the scene assets from the rig geometry: the mat texture (grid, tape square, numbered stickers), the mat
USD, and the two bowl USDs. Runs on the Mac (`lerobot` env: Pillow + matplotlib's DejaVu font). Outputs go to
`sim/so101_blocks/assets/` and are committed, so the simulator never regenerates them.
    python sim/so101_blocks/make_assets.py
"""
import math, os
from PIL import Image, ImageDraw, ImageFont
import rig

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "assets")
os.makedirs(OUT, exist_ok=True)
PPM = 4000                                   # texture pixels per metre (4 px/mm)
FONT = os.path.expanduser("~/miniforge3/envs/lerobot/lib/python3.12/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf")

DX, DY = rig.MAT_SIZE                        # mat extent along x (depth) and y (width)
X_MIN, X_MAX = rig.MAT_CENTER[0] - DX / 2, rig.MAT_CENTER[0] + DX / 2
Y_MIN, Y_MAX = rig.MAT_CENTER[1] - DY / 2, rig.MAT_CENTER[1] + DY / 2
W, H = int(round(DY * PPM)), int(round(DX * PPM))   # image columns run along -y (image left = +y), rows along -x (top = far edge)

def to_px(x: float, y: float) -> tuple[float, float]:
    return (Y_MAX - y) * PPM, (X_MAX - x) * PPM

def mat_texture() -> str:
    im = Image.new("RGB", (W, H), rig.MAT_RGB)
    d = ImageDraw.Draw(im)
    # 1 cm grid, brighter 5 cm lines, drawn on the whole mat (origin of the printed grid at the mat corner)
    for i in range(0, int(round(DY * 100)) + 1):
        c = round(i / 100 * PPM); major = i % 5 == 0
        d.line([(c, 0), (c, H)], fill=rig.MAT_GRID_RGB if major else tuple(int(v * 0.75) for v in rig.MAT_GRID_RGB), width=3 if major else 2)
    for i in range(0, int(round(DX * 100)) + 1):
        r = round(i / 100 * PPM); major = i % 5 == 0
        d.line([(0, r), (W, r)], fill=rig.MAT_GRID_RGB if major else tuple(int(v * 0.75) for v in rig.MAT_GRID_RGB), width=3 if major else 2)
    # two 45 degree guide lines through the mat centre, like the printed cutting mat
    cx, cy = to_px(*rig.MAT_CENTER)
    for s in (1, -1):
        d.line([(cx - H, cy - s * H), (cx + H, cy + s * H)], fill=rig.MAT_GRID_RGB, width=2)
    # tape square: outer edge minus tape width
    half, tw = rig.TAPE_OUTER / 2, rig.TAPE_WIDTH
    tx, ty = rig.TAPE_CENTER
    outer = [to_px(tx + half, ty + half), to_px(tx - half, ty - half)]   # (far-left corner) .. (near-right corner) in image coords
    d.rectangle([min(outer[0][0], outer[1][0]), min(outer[0][1], outer[1][1]), max(outer[0][0], outer[1][0]), max(outer[0][1], outer[1][1])], outline=(18, 18, 20), width=int(round(tw * PPM)))
    # stickers with numbers
    font = ImageFont.truetype(FONT, int(0.013 * PPM))
    r = rig.STICKER_R * PPM
    for k, (x, y) in rig.STICKERS.items():
        u, v = to_px(x, y)
        fill = (225, 225, 215) if k in rig.STICKER_WHITE else rig.STICKER_RGB
        d.ellipse([u - r, v - r, u + r, v + r], fill=fill)
        d.text((u, v), str(k), fill=(30, 30, 30), font=font, anchor="mm")
    path = os.path.join(OUT, "mat.png")
    im.save(path, optimize=True)
    return path

def mat_texture_photo(frames_glob: str = os.path.join(HERE, "..", "02_scene", "real_episodes", "first_*.jpg")) -> str:
    """The mat as the real overhead camera sees it: a per-pixel median over the real episodes' first frames (the block
    moves between episodes, so it vanishes; the arm and bowls are static and are excluded by region), warped into the
    texture's frame through rig.px_to_world, with the mat outside the camera's view tiled from a clean patch. Real
    tape, real stickers, real grid, real colours: the appearance the policy will see on the arm."""
    import glob
    import numpy as np
    files = sorted(glob.glob(frames_glob))
    assert len(files) >= 10, f"need the real first frames (export_grasps.py), found {len(files)}"
    stack = np.stack([np.asarray(Image.open(f).convert("RGB")) for f in files]).astype(np.uint8)
    med = np.median(stack, axis=0).astype(np.uint8)
    # a clean mat patch (grid only): the strip above the zone, between the bowls, tiled over the whole texture
    patch = med[4:72, 200:460]
    ppx = PPM / (rig.PX_PER_CM * 100)                       # texture pixels per image pixel
    patch_big = np.asarray(Image.fromarray(patch).resize((int(patch.shape[1] * ppx), int(patch.shape[0] * ppx)), Image.BICUBIC))
    reps = (H // patch_big.shape[0] + 2, W // patch_big.shape[1] + 2, 1)
    tex = np.tile(patch_big, reps)[:H, :W].copy()
    # the taped zone, tape included, pasted at its true place (the bowls and the arm's base lie outside these pixels)
    u0, u1, v0, v1 = 186, 474, 76, 355
    region = med[v0:v1, u0:u1].copy()
    # the arm at rest reaches into the zone's bottom rows in every real frame (image u 286-376, v 330-355): replace those
    # pixels with the same rows from further right, plain tape band and grid, so the arm is not baked into the mat
    region[330 - v0:339 - v0, 286 - u0:376 - u0] = region[330 - v0:339 - v0, 380 - u0:470 - u0]   # mat rows: the same rows further right
    region[339 - v0:355 - v0, 286 - u0:376 - u0] = np.array([18, 18, 20], np.uint8)                # tape rows: the tape's own colour
    x_top, y_left = rig.px_to_world(u0, v0)                # world coords of the region's image top-left corner
    col0, row0 = to_px(x_top, y_left)                       # texture position of that corner
    rw, rh = int(round((u1 - u0) * ppx)), int(round((v1 - v0) * ppx))
    region_big = np.asarray(Image.fromarray(region).resize((rw, rh), Image.BICUBIC))
    r0, c0 = int(round(row0)), int(round(col0))
    tex[r0:r0 + rh, c0:c0 + rw] = region_big[: max(0, min(rh, H - r0)), : max(0, min(rw, W - c0))]
    path = os.path.join(OUT, "mat.png")
    Image.fromarray(tex).save(path, optimize=True)
    return path


def mat_usda() -> str:
    """A thin box, top face at z=0 carrying the texture; triangle-mesh static collider."""
    t = rig.MAT_THICKNESS
    hx, hy = DX / 2, DY / 2
    P = [(-hx, -hy, -t), (hx, -hy, -t), (hx, hy, -t), (-hx, hy, -t), (-hx, -hy, 0), (hx, -hy, 0), (hx, hy, 0), (-hx, hy, 0)]
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]   # bottom, top, sides (outward normals)
    idx = [i for f in faces for i in f]
    # st per face-vertex: only the top face is mapped to the image; u = (y_max - y)/DY, v = (x - x_min)/DX (st origin = image bottom-left)
    st = []
    for f in faces:
        for i in f:
            x, y, z = P[i]
            st.append(((hy - y) / DY, (x + hx) / DX) if z == 0 else (0.0, 0.0))
    pts = ", ".join(f"({x:.4f}, {y:.4f}, {z:.4f})" for x, y, z in P)
    sts = ", ".join(f"({u:.4f}, {v:.4f})" for u, v in st)
    usda = f'''#usda 1.0
(
    defaultPrim = "Mat"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Mat"
{{
    def Mesh "Mesh" (
        prepend apiSchemas = ["MaterialBindingAPI", "PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
    )
    {{
        float3[] extent = [({-hx:.4f}, {-hy:.4f}, {-t:.4f}), ({hx:.4f}, {hy:.4f}, 0)]
        int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
        int[] faceVertexIndices = [{", ".join(map(str, idx))}]
        point3f[] points = [{pts}]
        texCoord2f[] primvars:st = [{sts}] (
            interpolation = "faceVarying"
        )
        uniform token subdivisionScheme = "none"
        uniform token physics:approximation = "convexHull"
        bool physics:collisionEnabled = 1
        rel material:binding = </Mat/Looks/MatMaterial>
    }}
    def Scope "Looks"
    {{
        def Material "MatMaterial"
        {{
            token outputs:surface.connect = </Mat/Looks/MatMaterial/Surface.outputs:surface>
            def Shader "Surface"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor.connect = </Mat/Looks/MatMaterial/Texture.outputs:rgb>
                float inputs:roughness = 0.8
                float inputs:metallic = 0
                token outputs:surface
            }}
            def Shader "Reader"
            {{
                uniform token info:id = "UsdPrimvarReader_float2"
                token inputs:varname = "st"
                float2 outputs:result
            }}
            def Shader "Texture"
            {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @./mat.png@
                float4 inputs:scale = (1, 1, 1, 1)
                float4 inputs:bias = (0, 0, 0, 0)
                float2 inputs:st.connect = </Mat/Looks/MatMaterial/Reader.outputs:result>
                token inputs:wrapS = "clamp"
                token inputs:wrapT = "clamp"
                float3 outputs:rgb
            }}
        }}
    }}
}}
'''
    path = os.path.join(OUT, "mat.usda")
    open(path, "w").write(usda)
    return path

def bowl_usda(name: str, rgb: tuple[int, int, int]) -> str:
    """Solid of revolution: a bowl with a base, flared wall and rounded rim; triangle-mesh static collider (concave is fine
    for a static body). Profile points are (radius, height)."""
    R, Ri, Rb, Hh = rig.BOWL_OUTER_R, rig.BOWL_INNER_R, rig.BOWL_BASE_R, rig.BOWL_HEIGHT
    profile = [(Rb, 0.0), (Rb + 0.004, 0.004), (R - 0.004, Hh * 0.55), (R, Hh), (R - 0.002, Hh + 0.002), (Ri, Hh), (Ri - 0.003, Hh * 0.55), (Rb - 0.004, 0.008), (Rb * 0.6, 0.006)]
    N = 48
    pts = []
    for (r, z) in profile:
        for i in range(N):
            a = 2 * math.pi * i / N
            pts.append((r * math.cos(a), r * math.sin(a), z))
    bottom_c = len(pts); pts.append((0.0, 0.0, 0.0))
    top_c = len(pts); pts.append((0.0, 0.0, profile[-1][1]))
    counts, idx = [], []
    for k in range(len(profile) - 1):
        for i in range(N):
            j = (i + 1) % N
            a, b, c, d = k * N + i, k * N + j, (k + 1) * N + j, (k + 1) * N + i
            counts.append(4); idx += [a, b, c, d]
    for i in range(N):                                       # bottom cap (outward = down)
        j = (i + 1) % N; counts.append(3); idx += [bottom_c, j, i]
    last = (len(profile) - 1) * N
    for i in range(N):                                       # inner floor cap (outward = up)
        j = (i + 1) % N; counts.append(3); idx += [top_c, last + i, last + j]
    ptxt = ", ".join(f"({x:.5f}, {y:.5f}, {z:.5f})" for x, y, z in pts)
    col = tuple(round(v / 255, 3) for v in rgb)
    usda = f'''#usda 1.0
(
    defaultPrim = "Bowl"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Bowl"
{{
    def Mesh "Mesh" (
        prepend apiSchemas = ["MaterialBindingAPI", "PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
    )
    {{
        float3[] extent = [({-R:.4f}, {-R:.4f}, 0), ({R:.4f}, {R:.4f}, {Hh + 0.002:.4f})]
        int[] faceVertexCounts = [{", ".join(map(str, counts))}]
        int[] faceVertexIndices = [{", ".join(map(str, idx))}]
        point3f[] points = [{ptxt}]
        uniform token subdivisionScheme = "none"
        uniform token physics:approximation = "none"
        bool physics:collisionEnabled = 1
        rel material:binding = </Bowl/Looks/BowlMaterial>
    }}
    def Scope "Looks"
    {{
        def Material "BowlMaterial"
        {{
            token outputs:surface.connect = </Bowl/Looks/BowlMaterial/Surface.outputs:surface>
            def Shader "Surface"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = ({col[0]}, {col[1]}, {col[2]})
                float inputs:roughness = 0.6
                float inputs:metallic = 0
                token outputs:surface
            }}
        }}
    }}
}}
'''
    path = os.path.join(OUT, f"bowl_{name}.usda")
    open(path, "w").write(usda)
    return path

if __name__ == "__main__":
    import sys
    if "--drawn" in sys.argv:
        print(mat_texture(), f"{W}x{H}")
    else:
        print(mat_texture_photo(), f"{W}x{H} (photo-based; --drawn for the synthetic grid)")
    print(mat_usda())
    for name, (_, _, rgb) in rig.BOWLS.items():
        print(bowl_usda(name, rgb))
