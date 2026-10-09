"""供桌面主进程管理的单实例服务，端口直接由系统分配。"""

import json
import os
import socket
import sys
import threading
import time

import uvicorn


def serve():
    if os.environ.get("QUIETTUNE_TRACE"):
        import faulthandler
        faulthandler.dump_traceback_later(10)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    listener.setblocking(False)
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config("backend.app.main:app", host="127.0.0.1", port=port,
                                          access_log=False, log_level="warning"))

    def watch_owner():
        # Windows 的阻塞标准输入会干扰科学计算 DLL 初始化，先探测管道再读取。
        if sys.platform == "win32":
            import ctypes
            import msvcrt
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            peek = kernel.PeekNamedPipe
            peek.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_void_p,
                             ctypes.POINTER(ctypes.c_ulong), ctypes.c_void_p]
            peek.restype = ctypes.c_int
            handle = msvcrt.get_osfhandle(sys.stdin.fileno())
            pending = b""
            while not server.should_exit:
                available = ctypes.c_ulong()
                if not peek(handle, None, 0, None, ctypes.byref(available), None):
                    break
                if available.value:
                    pending += os.read(sys.stdin.fileno(), min(available.value, 4096))
                    if b"shutdown\n" in pending.replace(b"\r\n", b"\n"):
                        break
                    pending = pending[-4096:]
                time.sleep(0.2)
        else:
            for line in sys.stdin:
                if line.strip() == "shutdown":
                    break
        server.should_exit = True

    threading.Thread(target=watch_owner, daemon=True).start()
    print(json.dumps({"type": "quiettune-service", "port": port}), flush=True)
    try:
        server.run(sockets=[listener])
    finally:
        listener.close()


if __name__ == "__main__":
    serve()
