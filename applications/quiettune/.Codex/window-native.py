"""仅对测试明确提供的窗口执行原生命中检查、系统命令和窗口截图。"""

import ctypes
import json
import sys
from ctypes import wintypes

from PIL import Image

user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
user32.IsWindow.argtypes = [wintypes.HWND]
user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.SendMessageW.restype = wintypes.LPARAM
user32.GetDC.argtypes = [wintypes.HWND]
user32.GetDC.restype = wintypes.HDC
user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
user32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
gdi32.CreateCompatibleDC.restype = wintypes.HDC
gdi32.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
gdi32.SelectObject.restype = wintypes.HGDIOBJ
gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
gdi32.DeleteDC.argtypes = [wintypes.HDC]


class BitmapHeader(ctypes.Structure):
    _fields_ = [("size", wintypes.DWORD), ("width", wintypes.LONG), ("height", wintypes.LONG),
                ("planes", wintypes.WORD), ("bit_count", wintypes.WORD), ("compression", wintypes.DWORD),
                ("size_image", wintypes.DWORD), ("x_resolution", wintypes.LONG), ("y_resolution", wintypes.LONG),
                ("colors_used", wintypes.DWORD), ("colors_important", wintypes.DWORD)]


gdi32.GetDIBits.argtypes = [wintypes.HDC, wintypes.HBITMAP, wintypes.UINT, wintypes.UINT, ctypes.c_void_p,
                          ctypes.POINTER(BitmapHeader), wintypes.UINT]


def capture(hwnd, destination):
    rectangle = wintypes.RECT()
    assert user32.GetWindowRect(hwnd, ctypes.byref(rectangle)), "读取窗口位置失败。"
    width, height = rectangle.right - rectangle.left, rectangle.bottom - rectangle.top
    screen_dc = user32.GetDC(None)
    memory_dc = gdi32.CreateCompatibleDC(screen_dc)
    bitmap = gdi32.CreateCompatibleBitmap(screen_dc, width, height)
    previous = gdi32.SelectObject(memory_dc, bitmap)
    try:
        assert user32.PrintWindow(hwnd, memory_dc, 2), "原生窗口截图失败。"
        header = BitmapHeader(ctypes.sizeof(BitmapHeader), width, -height, 1, 32, 0, width * height * 4, 0, 0, 0, 0)
        pixels = ctypes.create_string_buffer(width * height * 4)
        assert gdi32.GetDIBits(memory_dc, bitmap, 0, height, pixels, ctypes.byref(header), 0) == height
        picture = Image.frombuffer("RGB", (width, height), pixels, "raw", "BGRX", 0, 1)
        assert picture.getextrema() != ((0, 0), (0, 0), (0, 0)), "截图为空，不能作为视觉验收证据。"
        picture.save(destination)
    finally:
        gdi32.SelectObject(memory_dc, previous)
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(memory_dc)
        user32.ReleaseDC(None, screen_dc)


command, handle = sys.argv[1], int(sys.argv[2])
assert user32.IsWindow(handle), "测试窗口已经关闭。"
if command == "inspect":
    specification = json.loads(sys.argv[3])
    rectangle = wintypes.RECT()
    user32.GetClientRect(handle, ctypes.byref(rectangle))
    origin = wintypes.POINT(0, 0)
    user32.ClientToScreen(handle, ctypes.byref(origin))
    scale_x = rectangle.right / specification["width"]
    scale_y = rectangle.bottom / specification["height"]
    hits = {}
    for name, point in specification["points"].items():
        x, y = round(origin.x + point["x"] * scale_x), round(origin.y + point["y"] * scale_y)
        parameter = ((y & 0xffff) << 16) | (x & 0xffff)
        hits[name] = user32.SendMessageW(handle, 0x0084, 0, parameter)
    if specification.get("capture"):
        capture(handle, specification["capture"])
    print(json.dumps({"hits": hits, "scale": [scale_x, scale_y], "client_size": [rectangle.right, rectangle.bottom]}))
elif command == "system":
    actions = {"minimize": 0xF020, "maximize": 0xF030, "restore": 0xF120, "close": 0xF060}
    user32.SendMessageW(handle, 0x0112, actions[sys.argv[3]], 0)
    print(json.dumps({"action": sys.argv[3]}))
else:
    raise SystemExit("未知原生验证操作。")
