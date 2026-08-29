#!/bin/bash

# 默认值
cmake_args=""

# 显示帮助信息函数
show_help() {
    cat << EOF
用法: $0 [选项]

选项(只支持一次调用其中一个选项):
    --debug         启用调试模式构建
    --mssanitizer   启用MSSanitizer构建
    --simulator [SOC_VERSION]  启用仿真调优模式构建；设备目录名可作为第二参数传入，或事先 export SOC_VERSION（如 Ascend910B3、Ascend950PR_9589）
    --onboard       启用上板调优模式构建
    -h, --help      显示此帮助信息

示例:
    $0
    $0 --debug
    $0 --simulator Ascend910B3
EOF
    exit 0
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        --debug)
            cmake_args="-DENABLE_DEBUG=ON"
            break
            ;;
        --mssanitizer)
            cmake_args="-DENABLE_MS_SANITIZER=ON"
            break
            ;;
        --simulator)
            cmake_args="-DENABLE_SIMULATOR=ON -DASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
            if [[ $# -gt 1 ]] && [[ $2 != --* ]]; then
                cmake_args="$cmake_args -DSOC_VERSION=$2"
                break
            else
                cmake_args="$cmake_args -DSOC_VERSION=$SOC_VERSION"
                break
            fi
            ;;
        --onboard | --prof_onboard)
            cmake_args="-DENABLE_ON_BOARD=ON"
            break
            ;;
        *)
            break
            ;;
    esac
done


# 设置 CMAKE_PREFIX_PATH
export CMAKE_PREFIX_PATH=${ASCEND_TOOLKIT_HOME}/compiler/tikcpp/ascendc_kernel_cmake:$CMAKE_PREFIX_PATH
# 创建并进入构建目录
rm -rf ./build
mkdir -p build
cd build
# 配置和编译
cmake .. ${cmake_args}
make -j VERBOSE=1
