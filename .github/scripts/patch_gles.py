#!/usr/bin/env python3
"""
Make RSDKv5's iOS build draw with OpenGL ES 2 instead of SDL's Metal renderer,
so the video shaders (CRT-Yeetron, CRT-Yee64, Clean, YUV video...) work.

Usage:  python3 patch_gles.py <repo root>      (the folder holding RSDKv5.xcodeproj)

It patches three files in place (idempotent, safe to run twice):
  RSDKv5/RSDK/Graphics/SDL2/SDL2RenderDevice.hpp
  RSDKv5/RSDK/Graphics/SDL2/SDL2RenderDevice.cpp
  RSDKv5/RSDK/Core/RetroEngine.hpp           (adds the OpenGLES headers)

Shader sources are looked up in "Data/Shaders/OGL/" first (Data.rsdk, loose
files, mods) and then in "Shaders/OGL/" inside the app bundle, which the build
workflow fills from RSDKv5/Shaders/OGL.
"""
import os
import sys

MARK = "gles-render-patch"


def fail(msg):
    sys.exit("patch_gles.py: " + msg)


def must_replace(src, old, new, what):
    if src.count(old) != 1:
        fail("could not find %s exactly once - the source changed, the patch needs updating" % what)
    return src.replace(old, new, 1)


def mask_line(line):
    """Blank out string/char literals and comments so their braces are ignored."""
    out = []
    i, n = 0, len(line)
    while i < n:
        c = line[i]
        if c == "/" and i + 1 < n and line[i + 1] == "/":
            out.append(" " * (n - i))
            break
        if c == "/" and i + 1 < n and line[i + 1] == "*":
            j = line.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append(" " * (j - i))
            i = j
            continue
        if c == '"' or c == "'":
            quote = c
            out.append(" ")
            i += 1
            while i < n and line[i] != quote:
                if line[i] == "\\" and i + 1 < n:
                    out.append("  ")
                    i += 2
                    continue
                out.append(" ")
                i += 1
            out.append(" ")
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def scan_braces(text, start=0, stop_at_close=False):
    """
    Preprocessor-aware brace scanner: inside '#if ... #else ... #endif' only the first
    branch is counted (the engine opens a '{' in both branches of some '#if's).
    Returns the end offset of the block that opens first, or the net depth if stop_at_close is False.
    """
    stack = []  # one flag per open #if: True once we are inside its #else/#elif part
    depth = 0
    seen = False
    pos = start
    for line in text[start:].split("\n"):
        stripped = line.strip()
        if stripped.startswith("#"):
            word = stripped[1:].strip().split("(")[0].split()
            word = word[0] if word else ""
            if word in ("if", "ifdef", "ifndef"):
                stack.append(False)
            elif word in ("else", "elif"):
                if stack:
                    stack[-1] = True
            elif word == "endif":
                if stack:
                    stack.pop()
        elif not any(stack):
            for i, ch in enumerate(mask_line(line)):
                if ch == "{":
                    depth += 1
                    seen = True
                elif ch == "}":
                    depth -= 1
                    if stop_at_close and seen and depth == 0:
                        return pos + i + 1
        pos += len(line) + 1
    if stop_at_close:
        return -1
    return depth


def func_span(src, prefix):
    """Return (start, end) of the function whose definition starts with prefix."""
    if src.count(prefix) != 1:
        fail("function not found exactly once: " + prefix)
    start = src.index(prefix)
    end = scan_braces(src, start, stop_at_close=True)
    if end < 0:
        fail("unbalanced braces in " + prefix)
    return start, end


def replace_func(src, prefix, new):
    s, e = func_span(src, prefix)
    return src[:s] + new.strip("\n") + src[e:]


# ---------------------------------------------------------------------------
# RetroEngine.hpp: make the OpenGL ES headers visible everywhere
# ---------------------------------------------------------------------------
ENGINE_HEADER_INJECT = r'''// gles-render-patch
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE
#define GLES_SILENCE_DEPRECATION 1
#include <OpenGLES/ES2/gl.h>
#include <OpenGLES/ES2/glext.h>
#endif

'''

# ---------------------------------------------------------------------------
# SDL2RenderDevice.cpp replacements
# ---------------------------------------------------------------------------
TOP_BLOCK = r'''// ===== OpenGL ES render path (gles-render-patch) =====
SDL_GLContext RenderDevice::glContext = nullptr;
GLuint RenderDevice::screenTextures[SCREEN_COUNT];
GLuint RenderDevice::VBO           = 0;
uint32 *RenderDevice::videoBuffer  = nullptr;
bool32 RenderDevice::isInitialized = false;

#ifndef GL_BGRA_EXT
#define GL_BGRA_EXT 0x80E1
#endif

#define _GLESVERSION                                                                                                                                 \
    "#version 100\n#extension GL_OES_standard_derivatives : enable\n#define in_V attribute\n#define out varying\n#define in_F varying\n"

#if RETRO_REV02
#define _GLESDEFINE "#define RETRO_REV02 (1)\n"
#else
#define _GLESDEFINE "\n"
#endif

// video textures are uploaded as BGRA bytes: Y in red, U in green, V in blue
#define _YOFF 16
#define _UOFF 8
#define _VOFF 0

static char _glVPrecision[30];
static char _glFPrecision[30];
static float _vpX = 0, _vpY = 0, _vpW = 0, _vpH = 0;

static const GLchar *gles_backupVertex = R"aa(
in_V vec3 in_pos;
in_V vec2 in_UV;
out vec4 ex_color;
out vec2 ex_UV;

void main()
{
    gl_Position = vec4(in_pos, 1.0);
    ex_color    = vec4(1.0, 1.0, 1.0, 1.0);
    ex_UV       = in_UV;
}
)aa";

static const GLchar *gles_backupFragment = R"aa(
in_F vec2 ex_UV;
in_F vec4 ex_color;

uniform sampler2D texDiffuse;

void main()
{
    gl_FragColor = texture2D(texDiffuse, ex_UV);
}
)aa";

// Reads a shader source file. Looks in the engine's data first (Data.rsdk, loose Data folder, mods),
// then in the shaders bundled inside the app. Returns a malloc'd, NUL terminated buffer or NULL.
static char *gles_ReadShaderSource(const char *name)
{
    char path[0x200];
    FileInfo info;

    sprintf_s(path, sizeof(path), "Data/Shaders/OGL/%s", name);
    InitFileInfo(&info);
    if (LoadFile(&info, path, FMODE_RB)) {
        char *buffer = (char *)malloc(info.fileSize + 1);
        ReadBytes(&info, buffer, info.fileSize);
        buffer[info.fileSize] = 0;
        CloseFile(&info);
        return buffer;
    }

    char *base = SDL_GetBasePath();
    if (base) {
        sprintf_s(path, sizeof(path), "%sShaders/OGL/%s", base, name);
        SDL_free(base);

        FILE *f = fopen(path, "rb");
        if (f) {
            fseek(f, 0, SEEK_END);
            long size = ftell(f);
            fseek(f, 0, SEEK_SET);

            char *buffer = (char *)malloc(size + 1);
            size_t got   = fread(buffer, 1, size, f);
            buffer[got]  = 0;
            fclose(f);
            return buffer;
        }
    }

    PrintLog(PRINT_NORMAL, "Shader source not found: %s", name);
    return NULL;
}

@@VERTEX_TABLE@@
'''

FN_COPYFRAMEBUFFER = r'''
void RenderDevice::SetLinear(bool32 linear)
{
    if (!isInitialized)
        return;

    for (int32 i = 0; i < SCREEN_COUNT; ++i) {
        glBindTexture(GL_TEXTURE_2D, screenTextures[i]);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, linear ? GL_LINEAR : GL_NEAREST);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, linear ? GL_LINEAR : GL_NEAREST);
    }
}

void RenderDevice::CopyFrameBuffer()
{
    if (!isInitialized)
        return;

    for (int32 s = 0; s < videoSettings.screenCount; ++s) {
        glBindTexture(GL_TEXTURE_2D, screenTextures[s]);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, screens[s].pitch, SCREEN_YSIZE, GL_RGB, GL_UNSIGNED_SHORT_5_6_5, screens[s].frameBuffer);
    }
}
'''

FN_FLIPSCREEN = r'''
void RenderDevice::FlipScreen()
{
    if (!isInitialized)
        return;

    if (lastShaderID != videoSettings.shaderID) {
        lastShaderID = videoSettings.shaderID;

        SetLinear(shaderList[videoSettings.shaderID].linear);

        if (videoSettings.shaderSupport)
            glUseProgram(shaderList[videoSettings.shaderID].programID);
    }

    if (windowRefreshDelay > 0) {
        windowRefreshDelay--;
        if (!windowRefreshDelay)
            UpdateGameWindow();
        return;
    }

    glViewport((GLint)_vpX, (GLint)_vpY, (GLsizei)_vpW, (GLsizei)_vpH);

    // clearing ignores the viewport, so the pillarboxes stay black
    glClear(GL_COLOR_BUFFER_BIT);

    if (videoSettings.shaderSupport) {
        GLuint program = shaderList[videoSettings.shaderID].programID;
        glUniform2fv(glGetUniformLocation(program, "textureSize"), 1, &textureSize.x);
        glUniform2fv(glGetUniformLocation(program, "pixelSize"), 1, &pixelSize.x);
        glUniform2fv(glGetUniformLocation(program, "viewSize"), 1, &viewSize.x);
        glUniform1f(glGetUniformLocation(program, "screenDim"), videoSettings.dimMax * videoSettings.dimPercent);
    }

    int32 startVert = 0;
    switch (videoSettings.screenCount) {
        default:
        case 0:
#if RETRO_REV02
            startVert = 54;
#else
            startVert = 18;
#endif
            glBindTexture(GL_TEXTURE_2D, imageTexture);
            glDrawArrays(GL_TRIANGLES, startVert, 6);
            break;

        case 1:
            glBindTexture(GL_TEXTURE_2D, screenTextures[0]);
            glDrawArrays(GL_TRIANGLES, 0, 6);
            break;

        case 2:
#if RETRO_REV02
            startVert = startVertex_2P[0];
#else
            startVert = 6;
#endif
            glBindTexture(GL_TEXTURE_2D, screenTextures[0]);
            glDrawArrays(GL_TRIANGLES, startVert, 6);

#if RETRO_REV02
            startVert = startVertex_2P[1];
#else
            startVert = 12;
#endif
            glBindTexture(GL_TEXTURE_2D, screenTextures[1]);
            glDrawArrays(GL_TRIANGLES, startVert, 6);
            break;

#if RETRO_REV02
        case 3:
            glBindTexture(GL_TEXTURE_2D, screenTextures[0]);
            glDrawArrays(GL_TRIANGLES, startVertex_3P[0], 6);

            glBindTexture(GL_TEXTURE_2D, screenTextures[1]);
            glDrawArrays(GL_TRIANGLES, startVertex_3P[1], 6);

            glBindTexture(GL_TEXTURE_2D, screenTextures[2]);
            glDrawArrays(GL_TRIANGLES, startVertex_3P[2], 6);
            break;

        case 4:
            glBindTexture(GL_TEXTURE_2D, screenTextures[0]);
            glDrawArrays(GL_TRIANGLES, 30, 6);

            glBindTexture(GL_TEXTURE_2D, screenTextures[1]);
            glDrawArrays(GL_TRIANGLES, 36, 6);

            glBindTexture(GL_TEXTURE_2D, screenTextures[2]);
            glDrawArrays(GL_TRIANGLES, 42, 6);

            glBindTexture(GL_TEXTURE_2D, screenTextures[3]);
            glDrawArrays(GL_TRIANGLES, 48, 6);
            break;
#endif
    }

    SDL_GL_SwapWindow(window);
}
'''

FN_RELEASE = r'''
void RenderDevice::Release(bool32 isRefresh)
{
    if (isInitialized) {
        glDeleteTextures(SCREEN_COUNT, screenTextures);
        glDeleteTextures(1, &imageTexture);
        imageTexture = 0;

        glDeleteBuffers(1, &VBO);
        VBO = 0;

        if (videoBuffer)
            delete[] videoBuffer;
        videoBuffer = NULL;

        for (int32 i = 0; i < shaderCount; ++i) {
            if (shaderList[i].programID)
                glDeleteProgram(shaderList[i].programID);
            shaderList[i].programID = 0;
        }
    }

    isInitialized = false;

    if (!isRefresh) {
        shaderCount = 0;
#if RETRO_USE_MOD_LOADER
        userShaderCount = 0;
#endif

        if (displayInfo.displays)
            free(displayInfo.displays);
        displayInfo.displays = NULL;
    }

    if (!isRefresh && glContext)
        SDL_GL_DeleteContext(glContext);
    if (!isRefresh)
        glContext = nullptr;

    if (!isRefresh && window)
        SDL_DestroyWindow(window);

    if (!isRefresh)
        SDL_QuitSubSystem(SDL_INIT_VIDEO | SDL_INIT_EVENTS);

    if (!isRefresh) {
        if (scanlines)
            free(scanlines);
        scanlines = NULL;
    }
}
'''

FN_INITVERTEXBUFFER = r'''
void RenderDevice::InitVertexBuffer()
{
    RenderVertex vertBuffer[sizeof(rsdkGLESVertexBuffer) / sizeof(RenderVertex)];
    memcpy(vertBuffer, rsdkGLESVertexBuffer, sizeof(rsdkGLESVertexBuffer));

    float x = 0.5f / (float)viewSize.x;
    float y = 0.5f / (float)viewSize.y;

    // ignore the last 6 verts, they're scaled to the 1024x512 textures already!
    int32 vertCount = (RETRO_REV02 ? 60 : 24) - 6;
    for (int32 v = 0; v < vertCount; ++v) {
        RenderVertex *vertex = &vertBuffer[v];
        vertex->pos.x        = vertex->pos.x + x;
        vertex->pos.y        = vertex->pos.y - y;

        if (vertex->tex.x)
            vertex->tex.x = screens[0].size.x * (1.0 / textureSize.x);

        if (vertex->tex.y)
            vertex->tex.y = screens[0].size.y * (1.0 / textureSize.y);
    }

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(RenderVertex) * (!RETRO_REV02 ? 24 : 60), vertBuffer);
}
'''

FN_INITGRAPHICSAPI = r'''
bool RenderDevice::InitGraphicsAPI()
{
    isInitialized               = true;
    videoSettings.shaderSupport = true;

    viewSize.x = 0;
    viewSize.y = 0;

    if (videoSettings.windowed || !videoSettings.exclusiveFS) {
        if (videoSettings.windowed) {
            viewSize.x = videoSettings.windowWidth;
            viewSize.y = videoSettings.windowHeight;
        }
        else {
            viewSize.x = displayWidth[displayModeIndex];
            viewSize.y = displayHeight[displayModeIndex];
        }
    }
    else {
        int32 bufferWidth  = videoSettings.fsWidth;
        int32 bufferHeight = videoSettings.fsHeight;
        if (videoSettings.fsWidth <= 0 || videoSettings.fsHeight <= 0) {
            bufferWidth  = displayWidth[displayModeIndex];
            bufferHeight = displayHeight[displayModeIndex];
        }

        viewSize.x = bufferWidth;
        viewSize.y = bufferHeight;
    }

    SDL_SetWindowSize(window, viewSize.x, viewSize.y);
    SDL_SetWindowPosition(window, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);

    // From here on the view size is in real pixels: the drawable can be bigger than the window on retina screens.
    float ptsW = (float)viewSize.x;
    float ptsH = (float)viewSize.y;

    int32 drawW = 0, drawH = 0;
    SDL_GL_GetDrawableSize(window, &drawW, &drawH);
    if (drawW <= 0 || drawH <= 0) {
        drawW = (int32)ptsW;
        drawH = (int32)ptsH;
    }
    viewSize.x = drawW;
    viewSize.y = drawH;

    int32 maxPixHeight = 0;
#if !RETRO_USE_ORIGINAL_CODE
    int32 screenWidth = 0;
#endif
    for (int32 s = 0; s < 4; ++s) {
        if (videoSettings.pixHeight > maxPixHeight)
            maxPixHeight = videoSettings.pixHeight;

        screens[s].size.y = videoSettings.pixHeight;

        float viewAspect = (float)drawW / (float)drawH;
#if !RETRO_USE_ORIGINAL_CODE
        screenWidth = (int32)((viewAspect * videoSettings.pixHeight) + 3) & 0xFFFFFFFC;
#else
        int32 screenWidth = (int32)((viewAspect * videoSettings.pixHeight) + 3) & 0xFFFFFFFC;
#endif
        if (screenWidth < videoSettings.pixWidth)
            screenWidth = videoSettings.pixWidth;

#if !RETRO_USE_ORIGINAL_CODE
        if (customSettings.maxPixWidth && screenWidth > customSettings.maxPixWidth)
            screenWidth = customSettings.maxPixWidth;
#else
        if (screenWidth > DEFAULT_PIXWIDTH)
            screenWidth = DEFAULT_PIXWIDTH;
#endif

        memset(&screens[s].frameBuffer, 0, sizeof(screens[s].frameBuffer));
        SetScreenSize(s, screenWidth, screens[s].size.y);
    }

    pixelSize.x = screens[0].size.x;
    pixelSize.y = screens[0].size.y;

    videoSettings.pixWidth = pixelSize.x;

    // letterbox / pillarbox the game image inside the drawable
    float pixAspect  = (float)pixelSize.x / (float)pixelSize.y;
    float drawAspect = (float)drawW / (float)drawH;
    _vpX             = 0;
    _vpY             = 0;
    _vpW             = (float)drawW;
    _vpH             = (float)drawH;
    if (drawAspect <= pixAspect + 0.1f) {
        if (pixAspect - 0.1f > drawAspect) {
            float h = ((float)pixelSize.y / (float)pixelSize.x) * (float)drawW;
            _vpY    = ((float)drawH - h) * 0.5f;
            _vpH    = h;
            viewSize.y = h;
        }
    }
    else {
        float w = pixAspect * (float)drawH;
        _vpX    = ((float)drawW - w) * 0.5f;
        _vpW    = w;
        viewSize.x = w;
    }

#if !RETRO_USE_ORIGINAL_CODE
    bool32 smallTexture = screenWidth <= 512 && maxPixHeight <= 256;
#else
    bool32 smallTexture = maxPixHeight <= 256;
#endif
    if (smallTexture) {
        textureSize.x = 512.0;
        textureSize.y = 256.0;
    }
    else {
        textureSize.x = 1024.0;
        textureSize.y = 512.0;
    }

    // shaders need to know whether the GPU has high precision floats
    GLint range[2], precision;

    glGetShaderPrecisionFormat(GL_VERTEX_SHADER, GL_HIGH_FLOAT, range, &precision);
    strcpy(_glVPrecision, precision ? "precision highp float;\n" : "precision mediump float;\n");

    glGetShaderPrecisionFormat(GL_FRAGMENT_SHADER, GL_HIGH_FLOAT, range, &precision);
    strcpy(_glFPrecision, precision ? "precision highp float;\n" : "precision mediump float;\n");

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_DITHER);
    glDisable(GL_BLEND);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_CULL_FACE);

    // vertex buffer
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(RenderVertex) * (!RETRO_REV02 ? 24 : 60), NULL, GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(RenderVertex), 0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(RenderVertex), (void *)offsetof(RenderVertex, tex));
    glEnableVertexAttribArray(1);

    glViewport((GLint)_vpX, (GLint)_vpY, (GLsizei)_vpW, (GLsizei)_vpH);

    // textures
    glActiveTexture(GL_TEXTURE0);
    glGenTextures(SCREEN_COUNT, screenTextures);

    for (int32 i = 0; i < SCREEN_COUNT; ++i) {
        glBindTexture(GL_TEXTURE_2D, screenTextures[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, (GLsizei)textureSize.x, (GLsizei)textureSize.y, 0, GL_RGB, GL_UNSIGNED_SHORT_5_6_5, NULL);

        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }

    glGenTextures(1, &imageTexture);
    glBindTexture(GL_TEXTURE_2D, imageTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, RETRO_VIDEO_TEXTURE_W, RETRO_VIDEO_TEXTURE_H, 0, GL_BGRA_EXT, GL_UNSIGNED_BYTE, NULL);

    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    if (videoBuffer)
        delete[] videoBuffer;
    videoBuffer = new uint32[RETRO_VIDEO_TEXTURE_W * RETRO_VIDEO_TEXTURE_H];

    lastShaderID = -1;
    InitVertexBuffer();
    engine.inFocus          = 1;
    videoSettings.viewportX = 0;
    videoSettings.viewportY = 0;
    videoSettings.viewportW = 1.0 / ptsW;
    videoSettings.viewportH = 1.0 / ptsH;

    return true;
}
'''

FN_LOADSHADER = r'''
void RenderDevice::LoadShader(const char *fileName, bool32 linear)
{
    char fragName[0x100];

    for (int32 i = 0; i < shaderCount; ++i) {
        if (strcmp(shaderList[i].name, fileName) == 0)
            return;
    }

    if (shaderCount == SHADER_COUNT)
        return;

    ShaderEntry *shader = &shaderList[shaderCount];
    shader->linear      = linear;
    sprintf_s(shader->name, sizeof(shader->name), "%s", fileName);

    GLint success;
    char infoLog[0x1000];
    GLuint vert, frag;

    char *source = gles_ReadShaderSource("None.vs");
    if (!source)
        return;

    const GLchar *vchar[] = { _GLESVERSION, _GLESDEFINE, _glVPrecision, (const GLchar *)source };
    vert                  = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vert, 4, vchar, NULL);
    glCompileShader(vert);
    free(source);

    glGetShaderiv(vert, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vert, 0x1000, NULL, infoLog);
        PrintLog(PRINT_NORMAL, "Vertex shader compiling failed:\n%s", infoLog);
        glDeleteShader(vert);
        return;
    }

    sprintf_s(fragName, sizeof(fragName), "%s.fs", fileName);
    source = gles_ReadShaderSource(fragName);
    if (!source) {
        glDeleteShader(vert);
        return;
    }

    const GLchar *fchar[] = { _GLESVERSION, _GLESDEFINE, _glFPrecision, (const GLchar *)source };
    frag                  = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(frag, 4, fchar, NULL);
    glCompileShader(frag);
    free(source);

    glGetShaderiv(frag, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(frag, 0x1000, NULL, infoLog);
        PrintLog(PRINT_NORMAL, "Fragment shader compiling failed (%s):\n%s", fileName, infoLog);
        glDeleteShader(vert);
        glDeleteShader(frag);
        return;
    }

    shader->programID = glCreateProgram();
    glAttachShader(shader->programID, vert);
    glAttachShader(shader->programID, frag);

    glBindAttribLocation(shader->programID, 0, "in_pos");
    glBindAttribLocation(shader->programID, 1, "in_UV");

    glLinkProgram(shader->programID);
    glGetProgramiv(shader->programID, GL_LINK_STATUS, &success);
    glDeleteShader(vert);
    glDeleteShader(frag);
    if (!success) {
        glGetProgramInfoLog(shader->programID, 0x1000, NULL, infoLog);
        PrintLog(PRINT_NORMAL, "OpenGL shader linking failed:\n%s", infoLog);
        glDeleteProgram(shader->programID);
        shader->programID = 0;
        return;
    }

    shaderCount++;
}
'''

FN_INITSHADERS = r'''
bool RenderDevice::InitShaders()
{
    videoSettings.shaderSupport = true;
    int32 maxShaders            = 0;
    shaderCount                 = 0;

    LoadShader("None", false);
    LoadShader("Clean", true);
    LoadShader("CRT-Yeetron", true);
    LoadShader("CRT-Yee64", true);

#if RETRO_USE_MOD_LOADER
    // a place for mods to load custom shaders
    RunModCallbacks(MODCB_ONSHADERLOAD, NULL);
    userShaderCount = shaderCount;
#endif

    LoadShader("YUV-420", true);
    LoadShader("YUV-422", true);
    LoadShader("YUV-444", true);
    LoadShader("RGB-Image", true);
    maxShaders = shaderCount;

    // no shaders == no support: fall back to a plain textured quad
    if (!maxShaders) {
        ShaderEntry *shader         = &shaderList[0];
        videoSettings.shaderSupport = false;

        maxShaders  = 1;
        shaderCount = 1;

        GLint success;
        char infoLog[0x1000];

        GLuint vert, frag;
        const GLchar *vchar[] = { _GLESVERSION, _GLESDEFINE, _glVPrecision, gles_backupVertex };
        vert                  = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vert, 4, vchar, NULL);
        glCompileShader(vert);

        const GLchar *fchar[] = { _GLESVERSION, _GLESDEFINE, _glFPrecision, gles_backupFragment };
        frag                  = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(frag, 4, fchar, NULL);
        glCompileShader(frag);

        glGetShaderiv(vert, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(vert, 0x1000, NULL, infoLog);
            PrintLog(PRINT_NORMAL, "BACKUP vertex shader compiling failed:\n%s", infoLog);
        }

        glGetShaderiv(frag, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(frag, 0x1000, NULL, infoLog);
            PrintLog(PRINT_NORMAL, "BACKUP fragment shader compiling failed:\n%s", infoLog);
        }

        shader->programID = glCreateProgram();
        glAttachShader(shader->programID, vert);
        glAttachShader(shader->programID, frag);

        glBindAttribLocation(shader->programID, 0, "in_pos");
        glBindAttribLocation(shader->programID, 1, "in_UV");

        glLinkProgram(shader->programID);
        glDeleteShader(vert);
        glDeleteShader(frag);

        glUseProgram(shader->programID);

        shader->linear = videoSettings.windowed ? false : true;
    }

    videoSettings.shaderID = MAX(videoSettings.shaderID >= maxShaders ? 0 : videoSettings.shaderID, 0);
    SetLinear(shaderList[videoSettings.shaderID].linear || videoSettings.screenCount > 1);

    return true;
}
'''

FN_SETUPRENDERING = r'''
bool RenderDevice::SetupRendering()
{
    glContext = SDL_GL_CreateContext(window);
    if (!glContext) {
        PrintLog(PRINT_NORMAL, "ERROR: failed to create OpenGL ES context!\nerror msg: %s", SDL_GetError());
        return false;
    }

    SDL_GL_MakeCurrent(window, glContext);
    SDL_GL_SetSwapInterval(videoSettings.vsync ? 1 : 0);

    GetDisplays();

    if (!InitGraphicsAPI() || !InitShaders())
        return false;

    int32 size = videoSettings.pixWidth >= SCREEN_YSIZE ? videoSettings.pixWidth : SCREEN_YSIZE;
    scanlines  = (ScanlineInfo *)malloc(size * sizeof(ScanlineInfo));
    memset(scanlines, 0, size * sizeof(ScanlineInfo));

    videoSettings.windowState = WINDOWSTATE_ACTIVE;
    videoSettings.dimMax      = 1.0;
    videoSettings.dimPercent  = 1.0;

    return true;
}
'''

FN_GETWINDOWSIZE = r'''
void RenderDevice::GetWindowSize(int32 *width, int32 *height)
{
    if (!videoSettings.windowed) {
        int32 w = 0, h = 0;
        SDL_GL_GetDrawableSize(window, &w, &h);

        if (width)
            *width = w;

        if (height)
            *height = h;
    }
    else {
        int32 currentWindowDisplay = SDL_GetWindowDisplayIndex(window);

        SDL_DisplayMode display;
        SDL_GetCurrentDisplayMode(currentWindowDisplay, &display);

        if (width)
            *width = display.w;

        if (height)
            *height = display.h;
    }
}
'''

FN_SETUPIMAGETEXTURE = r'''
void RenderDevice::SetupImageTexture(int32 width, int32 height, uint8 *imagePixels)
{
    if (imagePixels && isInitialized) {
        glBindTexture(GL_TEXTURE_2D, imageTexture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_BGRA_EXT, GL_UNSIGNED_BYTE, imagePixels);
    }
}
'''

FN_YUV420 = r'''
void RenderDevice::SetupVideoTexture_YUV420(int32 width, int32 height, uint8 *yPlane, uint8 *uPlane, uint8 *vPlane, int32 strideY, int32 strideU,
                                            int32 strideV)
{
    if (!isInitialized)
        return;

    uint32 *pixels = videoBuffer;
    uint32 *preY   = pixels;
    int32 pitch    = RETRO_VIDEO_TEXTURE_W - width;
    if (videoSettings.shaderSupport) {
        for (int32 y = 0; y < height; ++y) {
            for (int32 x = 0; x < width; ++x) {
                *pixels++ = (yPlane[x] << _YOFF) | 0xFF000000;
            }

            pixels += pitch;
            yPlane += strideY;
        }

        pixels = preY;
        pitch  = RETRO_VIDEO_TEXTURE_W - (width >> 1);
        for (int32 y = 0; y < (height >> 1); ++y) {
            for (int32 x = 0; x < (width >> 1); ++x) {
                *pixels++ |= (vPlane[x] << _VOFF) | (uPlane[x] << _UOFF) | 0xFF000000;
            }

            pixels += pitch;
            uPlane += strideU;
            vPlane += strideV;
        }
    }
    else {
        // No shader support means no YUV support! at least use the brightness to show it in grayscale!
        for (int32 y = 0; y < height; ++y) {
            for (int32 x = 0; x < width; ++x) {
                int32 brightness = yPlane[x];
                *pixels++        = (brightness << 0) | (brightness << 8) | (brightness << 16) | 0xFF000000;
            }

            pixels += pitch;
            yPlane += strideY;
        }
    }

    glBindTexture(GL_TEXTURE_2D, imageTexture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, RETRO_VIDEO_TEXTURE_W, RETRO_VIDEO_TEXTURE_H, GL_BGRA_EXT, GL_UNSIGNED_BYTE, videoBuffer);
}
'''

FN_YUV422 = r'''
void RenderDevice::SetupVideoTexture_YUV422(int32 width, int32 height, uint8 *yPlane, uint8 *uPlane, uint8 *vPlane, int32 strideY, int32 strideU,
                                            int32 strideV)
{
    if (!isInitialized)
        return;

    uint32 *pixels = videoBuffer;
    uint32 *preY   = pixels;
    int32 pitch    = RETRO_VIDEO_TEXTURE_W - width;

    if (videoSettings.shaderSupport) {
        for (int32 y = 0; y < height; ++y) {
            for (int32 x = 0; x < width; ++x) {
                *pixels++ = (yPlane[x] << _YOFF) | 0xFF000000;
            }

            pixels += pitch;
            yPlane += strideY;
        }

        pixels = preY;
        pitch  = RETRO_VIDEO_TEXTURE_W - (width >> 1);
        for (int32 y = 0; y < height; ++y) {
            for (int32 x = 0; x < (width >> 1); ++x) {
                *pixels++ |= (vPlane[x] << _VOFF) | (uPlane[x] << _UOFF) | 0xFF000000;
            }

            pixels += pitch;
            uPlane += strideU;
            vPlane += strideV;
        }
    }
    else {
        for (int32 y = 0; y < height; ++y) {
            for (int32 x = 0; x < width; ++x) {
                int32 brightness = yPlane[x];
                *pixels++        = (brightness << 0) | (brightness << 8) | (brightness << 16) | 0xFF000000;
            }

            pixels += pitch;
            yPlane += strideY;
        }
    }

    glBindTexture(GL_TEXTURE_2D, imageTexture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, RETRO_VIDEO_TEXTURE_W, RETRO_VIDEO_TEXTURE_H, GL_BGRA_EXT, GL_UNSIGNED_BYTE, videoBuffer);
}
'''

FN_YUV444 = r'''
void RenderDevice::SetupVideoTexture_YUV444(int32 width, int32 height, uint8 *yPlane, uint8 *uPlane, uint8 *vPlane, int32 strideY, int32 strideU,
                                            int32 strideV)
{
    if (!isInitialized)
        return;

    uint32 *pixels = videoBuffer;
    int32 pitch    = RETRO_VIDEO_TEXTURE_W - width;
    if (videoSettings.shaderSupport) {
        for (int32 y = 0; y < height; ++y) {
            int32 pos1  = yPlane - vPlane;
            int32 pos2  = uPlane - vPlane;
            uint8 *pixV = vPlane;
            for (int32 x = 0; x < width; ++x) {
                *pixels++ = (pixV[0] << _VOFF) | (pixV[pos2] << _UOFF) | (pixV[pos1] << _YOFF) | 0xFF000000;
                pixV++;
            }

            pixels += pitch;
            yPlane += strideY;
            uPlane += strideU;
            vPlane += strideV;
        }
    }
    else {
        for (int32 y = 0; y < height; ++y) {
            for (int32 x = 0; x < width; ++x) {
                int32 brightness = yPlane[x];
                *pixels++        = (brightness << 0) | (brightness << 8) | (brightness << 16) | 0xFF000000;
            }

            pixels += pitch;
            yPlane += strideY;
        }
    }

    glBindTexture(GL_TEXTURE_2D, imageTexture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, RETRO_VIDEO_TEXTURE_W, RETRO_VIDEO_TEXTURE_H, GL_BGRA_EXT, GL_UNSIGNED_BYTE, videoBuffer);
}
'''

WINDOW_FLAGS_OLD = "    int flags = SDL_WINDOW_METAL;"
WINDOW_FLAGS_NEW = r'''    int flags = SDL_WINDOW_OPENGL;
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_ES);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 0);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 0);'''


def patch_cpp(path, vertex_table):
    src = open(path, encoding="utf-8").read()
    if MARK in src:
        print("SDL2RenderDevice.cpp: already patched")
        return
    depth_before = scan_braces(src)

    src = must_replace(src, "SDL_Texture *RenderDevice::screenTexture[SCREEN_COUNT];\n",
                       TOP_BLOCK.replace("@@VERTEX_TABLE@@", vertex_table), "screenTexture definition")
    src = must_replace(src, "SDL_Texture *RenderDevice::imageTexture = nullptr;", "GLuint RenderDevice::imageTexture = 0;",
                       "imageTexture definition")
    src = must_replace(src, WINDOW_FLAGS_OLD, WINDOW_FLAGS_NEW, "window flags (SDL_WINDOW_METAL)")

    src = replace_func(src, "void RenderDevice::CopyFrameBuffer()", FN_COPYFRAMEBUFFER)
    src = replace_func(src, "void RenderDevice::FlipScreen()", FN_FLIPSCREEN)
    src = replace_func(src, "void RenderDevice::Release(bool32 isRefresh)", FN_RELEASE)
    src = replace_func(src, "void RenderDevice::InitVertexBuffer()", FN_INITVERTEXBUFFER)
    src = replace_func(src, "bool RenderDevice::InitGraphicsAPI()", FN_INITGRAPHICSAPI)
    src = replace_func(src, "void RenderDevice::LoadShader(", FN_LOADSHADER)
    src = replace_func(src, "bool RenderDevice::InitShaders()", FN_INITSHADERS)
    src = replace_func(src, "bool RenderDevice::SetupRendering()", FN_SETUPRENDERING)
    src = replace_func(src, "void RenderDevice::GetWindowSize(", FN_GETWINDOWSIZE)
    src = replace_func(src, "void RenderDevice::SetupImageTexture(", FN_SETUPIMAGETEXTURE)
    src = replace_func(src, "void RenderDevice::SetupVideoTexture_YUV420(", FN_YUV420)
    src = replace_func(src, "void RenderDevice::SetupVideoTexture_YUV422(", FN_YUV422)
    src = replace_func(src, "void RenderDevice::SetupVideoTexture_YUV444(", FN_YUV444)

    if scan_braces(src) != depth_before:
        fail("brace balance changed after patching SDL2RenderDevice.cpp")
    open(path, "w", encoding="utf-8").write(src)
    print("SDL2RenderDevice.cpp: patched")


def patch_hpp(path):
    src = open(path, encoding="utf-8").read()
    if MARK in src:
        print("SDL2RenderDevice.hpp: already patched")
        return

    src = must_replace(src, "using ShaderEntry = ShaderEntryBase;",
                       "// gles-render-patch\nstruct ShaderEntry : public ShaderEntryBase {\n    GLuint programID;\n};", "ShaderEntry alias")
    src = must_replace(src, "    static SDL_Texture *screenTexture[SCREEN_COUNT];\n",
                       "    static SDL_GLContext glContext;\n    static GLuint screenTextures[SCREEN_COUNT];\n"
                       "    static GLuint VBO;\n    static uint32 *videoBuffer;\n    static bool32 isInitialized;\n"
                       "    static void SetLinear(bool32 linear);\n", "screenTexture declaration")
    src = must_replace(src, "    static SDL_Texture *imageTexture;", "    static GLuint imageTexture;", "imageTexture declaration")
    open(path, "w", encoding="utf-8").write(src)
    print("SDL2RenderDevice.hpp: patched")


def patch_engine_header(path):
    src = open(path, encoding="utf-8").read()
    if MARK in src:
        print("RetroEngine.hpp: already patched")
        return
    open(path, "w", encoding="utf-8").write(ENGINE_HEADER_INJECT + src)
    print("RetroEngine.hpp: patched")


VERTEX_TABLE = r'''#if RETRO_REV02
const RenderVertex rsdkGLESVertexBuffer[60] = {
    // 1 Screen (0)
    { { +1.0, -1.0,  1.0 }, 0xFFFFFFFF, {  1.0,  1.0 } },
    { { +1.0, +1.0,  1.0 }, 0xFFFFFFFF, {  1.0,  0.0 } },
    { { -1.0, +1.0,  1.0 }, 0xFFFFFFFF, {  0.0,  0.0 } },
    { { +1.0, -1.0,  1.0 }, 0xFFFFFFFF, {  1.0,  1.0 } },
    { { -1.0, -1.0,  1.0 }, 0xFFFFFFFF, {  0.0,  1.0 } },
    { { -1.0, +1.0,  1.0 }, 0xFFFFFFFF, {  0.0,  0.0 } },
    
    // 2 Screens - Bordered (Top Screen) (6)
    { { +0.5,  0.0,  1.0 }, 0xFFFFFFFF, {  1.0,  1.0 } },
    { { +0.5, +1.0,  1.0 }, 0xFFFFFFFF, {  1.0,  0.0 } },
    { { -0.5, +1.0,  1.0 }, 0xFFFFFFFF, {  0.0,  0.0 } },
    { { +0.5,  0.0,  1.0 }, 0xFFFFFFFF, {  1.0,  1.0 } },
    { { -0.5,  0.0,  1.0 }, 0xFFFFFFFF, {  0.0,  1.0 } },
    { { -0.5, +1.0,  1.0 }, 0xFFFFFFFF, {  0.0,  0.0 } },
    
    // 2 Screens - Bordered (Bottom Screen) (12)
    { { +0.5, -1.0,  1.0 }, 0xFFFFFFFF, {  1.0,  1.0 } },
    { { +0.5,  0.0,  1.0 }, 0xFFFFFFFF, {  1.0,  0.0 } },
    { { -0.5,  0.0,  1.0 }, 0xFFFFFFFF, {  0.0,  0.0 } },
    { { +0.5, -1.0,  1.0 }, 0xFFFFFFFF, {  1.0,  1.0 } },
    { { -0.5, -1.0,  1.0 }, 0xFFFFFFFF, {  0.0,  1.0 } },
    { { -0.5,  0.0,  1.0 }, 0xFFFFFFFF, {  0.0,  0.0 } },
    
    // 2 Screens - Stretched (Top Screen) (18)
    { { +1.0,  0.0,  1.0 }, 0xFFFFFFFF, {  1.0,  1.0 } },
    { { +1.0, +1.0,  1.0 }, 0xFFFFFFFF, {  1.0,  0.0 } },
    { { -1.0, +1.0,  1.0 }, 0xFFFFFFFF, {  0.0,  0.0 } },
    { { +1.0,  0.0,  1.0 }, 0xFFFFFFFF, {  1.0,  1.0 } },
    { { -1.0,  0.0,  1.0 }, 0xFFFFFFFF, {  0.0,  1.0 } },
    { { -1.0, +1.0,  1.0 }, 0xFFFFFFFF, {  0.0,  0.0 } },
  
    // 2 Screens - Stretched (Bottom Screen) (24)
    { { +1.0, -1.0,  1.0 }, 0xFFFFFFFF, {  1.0,  1.0 } },
    { { +1.0,  0.0,  1.0 }, 0xFFFFFFFF, {  1.0,  0.0 } },
    { { -1.0,  0.0,  1.0 }, 0xFFFFFFFF, {  0.0,  0.0 } },
    { { +1.0, -1.0,  1.0 }, 0xFFFFFFFF, {  1.0,  1.0 } },
    { { -1.0, -1.0,  1.0 }, 0xFFFFFFFF, {  0.0,  1.0 } },
    { { -1.0,  0.0,  1.0 }, 0xFFFFFFFF, {  0.0,  0.0 } },
    
    // 4 Screens (Top-Left) (30)
    { {  0.0,  0.0,  1.0 }, 0xFFFFFFFF, {  1.0,  1.0 } },
    { {  0.0, +1.0,  1.0 }, 0xFFFFFFFF, {  1.0,  0.0 } },
    { { -1.0, +1.0,  1.0 }, 0xFFFFFFFF, {  0.0,  0.0 } },
    { {  0.0,  0.0,  1.0 }, 0xFFFFFFFF, {  1.0,  1.0 } },
    { { -1.0,  0.0,  1.0 }, 0xFFFFFFFF, {  0.0,  1.0 } },
    { { -1.0, +1.0,  1.0 }, 0xFFFFFFFF, {  0.0,  0.0 } },

    // 4 Screens (Top-Right) (36)
    { { +1.0,  0.0,  1.0 }, 0xFFFFFFFF, {  1.0,  1.0 } },
    { { +1.0, +1.0,  1.0 }, 0xFFFFFFFF, {  1.0,  0.0 } },
    { {  0.0, +1.0,  1.0 }, 0xFFFFFFFF, {  0.0,  0.0 } },
    { { +1.0,  0.0,  1.0 }, 0xFFFFFFFF, {  1.0,  1.0 } },
    { {  0.0,  0.0,  1.0 }, 0xFFFFFFFF, {  0.0,  1.0 } },
    { {  0.0, +1.0,  1.0 }, 0xFFFFFFFF, {  0.0,  0.0 } },
    
    // 4 Screens (Bottom-Right) (42)
    { {  0.0, -1.0,  1.0 }, 0xFFFFFFFF, {  1.0,  1.0 } },
    { {  0.0,  0.0,  1.0 }, 0xFFFFFFFF, {  1.0,  0.0 } },
    { { -1.0,  0.0,  1.0 }, 0xFFFFFFFF, {  0.0,  0.0 } },
    { {  0.0, -1.0,  1.0 }, 0xFFFFFFFF, {  1.0,  1.0 } },
    { { -1.0, -1.0,  1.0 }, 0xFFFFFFFF, {  0.0,  1.0 } },
    { { -1.0,  0.0,  1.0 }, 0xFFFFFFFF, {  0.0,  0.0 } },
    
    // 4 Screens (Bottom-Left) (48)
    { { +1.0, -1.0,  1.0 }, 0xFFFFFFFF, {  1.0,  1.0 } },
    { { +1.0,  0.0,  1.0 }, 0xFFFFFFFF, {  1.0,  0.0 } },
    { {  0.0,  0.0,  1.0 }, 0xFFFFFFFF, {  0.0,  0.0 } },
    { { +1.0, -1.0,  1.0 }, 0xFFFFFFFF, {  1.0,  1.0 } },
    { {  0.0, -1.0,  1.0 }, 0xFFFFFFFF, {  0.0,  1.0 } },
    { {  0.0,  0.0,  1.0 }, 0xFFFFFFFF, {  0.0,  0.0 } },
    
    // Image/Video (54)
    { { +1.0, -1.0,  1.0 }, 0xFFFFFFFF, {  1.0,  1.0 } },
    { { +1.0, +1.0,  1.0 }, 0xFFFFFFFF, {  1.0,  0.0 } },
    { { -1.0, +1.0,  1.0 }, 0xFFFFFFFF, {  0.0,  0.0 } },
    { { +1.0, -1.0,  1.0 }, 0xFFFFFFFF, {  1.0,  1.0 } },
    { { -1.0, -1.0,  1.0 }, 0xFFFFFFFF, {  0.0,  1.0 } },
    { { -1.0, +1.0,  1.0 }, 0xFFFFFFFF, {  0.0,  0.0 } }
};
#else
const RenderVertex rsdkGLESVertexBuffer[24] =
{
    // 1 Screen (0)
    { { +1.0, -1.0,  1.0 }, 0xFFFFFFFF, {  1.0,  1.0 } },
    { { +1.0, +1.0,  1.0 }, 0xFFFFFFFF, {  1.0,  0.0 } },
    { { -1.0, +1.0,  1.0 }, 0xFFFFFFFF, {  0.0,  0.0 } },
    { { +1.0, -1.0,  1.0 }, 0xFFFFFFFF, {  1.0,  1.0 } },
    { { -1.0, -1.0,  1.0 }, 0xFFFFFFFF, {  0.0,  1.0 } },
    { { -1.0, +1.0,  1.0 }, 0xFFFFFFFF, {  0.0,  0.0 } },

    // 2 Screens - Stretched (Top Screen) (6)
    { { +1.0,  0.0,  1.0 }, 0xFFFFFFFF, {  1.0,  1.0 } },
    { { +1.0, +1.0,  1.0 }, 0xFFFFFFFF, {  1.0,  0.0 } },
    { { -1.0, +1.0,  1.0 }, 0xFFFFFFFF, {  0.0,  0.0 } },
    { { +1.0,  0.0,  1.0 }, 0xFFFFFFFF, {  1.0,  1.0 } },
    { { -1.0,  0.0,  1.0 }, 0xFFFFFFFF, {  0.0,  1.0 } },
    { { -1.0, +1.0,  1.0 }, 0xFFFFFFFF, {  0.0,  0.0 } },
  
    // 2 Screens - Stretched (Bottom Screen) (12)
    { { +1.0, -1.0,  1.0 }, 0xFFFFFFFF, {  1.0,  1.0 } },
    { { +1.0,  0.0,  1.0 }, 0xFFFFFFFF, {  1.0,  0.0 } },
    { { -1.0,  0.0,  1.0 }, 0xFFFFFFFF, {  0.0,  0.0 } },
    { { +1.0, -1.0,  1.0 }, 0xFFFFFFFF, {  1.0,  1.0 } },
    { { -1.0, -1.0,  1.0 }, 0xFFFFFFFF, {  0.0,  1.0 } },
    { { -1.0,  0.0,  1.0 }, 0xFFFFFFFF, {  0.0,  0.0 } },
  
    // Image/Video (18)
    { { +1.0, -1.0,  1.0 }, 0xFFFFFFFF, {  1.0,  1.0 } },
    { { +1.0, +1.0,  1.0 }, 0xFFFFFFFF, {  1.0,  0.0 } },
    { { -1.0, +1.0,  1.0 }, 0xFFFFFFFF, {  0.0,  0.0 } },
    { { +1.0, -1.0,  1.0 }, 0xFFFFFFFF, {  1.0,  1.0 } },
    { { -1.0, -1.0,  1.0 }, 0xFFFFFFFF, {  0.0,  1.0 } },
    { { -1.0, +1.0,  1.0 }, 0xFFFFFFFF, {  0.0,  0.0 } }
};
#endif'''


def main():
    if len(sys.argv) != 2:
        fail("usage: patch_gles.py <repo root>")
    root = sys.argv[1]
    gfx = os.path.join(root, "RSDKv5/RSDK/Graphics/SDL2")
    paths = {
        "cpp": os.path.join(gfx, "SDL2RenderDevice.cpp"),
        "hpp": os.path.join(gfx, "SDL2RenderDevice.hpp"),
        "engine": os.path.join(root, "RSDKv5/RSDK/Core/RetroEngine.hpp"),
    }
    for p in paths.values():
        if not os.path.isfile(p):
            fail("missing file: " + p)

    patch_hpp(paths["hpp"])
    patch_cpp(paths["cpp"], VERTEX_TABLE)
    patch_engine_header(paths["engine"])


if __name__ == "__main__":
    main()
