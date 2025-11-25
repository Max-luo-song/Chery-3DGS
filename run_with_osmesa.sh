    #!/bin/bash
    
    # 设置库路径，确保使用我们新编译的 Mesa
    export LD_LIBRARY_PATH=/opt/mesa-18.3.3/usr/local/lib:$LD_LIBRARY_PATH
    
    # 强制 PyOpenGL 使用 GLX 平台，这个平台通常会正确链接到 OSMesa
    export PYOPENGL_PLATFORM=glx
    
    # 执行传入给这个脚本的所有参数（也就是你的 python 命令）
    exec "$@"
    
