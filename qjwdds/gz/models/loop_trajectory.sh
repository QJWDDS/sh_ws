#!/bin/bash

# --- 配置参数 ---
TOPIC_X="/model/sptballoon/joint/x_axis_joint/0/cmd_pos"
TOPIC_Y="/model/sptballoon/joint/y_axis_joint/0/cmd_pos"
TOPIC_Z="/model/sptballoon/joint/z_axis_joint/0/cmd_pos"

BASE_SIZE=54      # 底边长
HEIGHT=40         # 总高度
STEP_SIZE=4       # 每次运动距离 4米
SLEEP_TIME=1.0    # 每次运动等待 1秒

# --- 函数：发布指令 ---
# 参数 $1: 话题, $2: 数值
publish_cmd() {
    gz topic -t "$1" -m gz.msgs.Double -p "data: $2"
}

# --- 主逻辑 ---

# 初始化
cur_x=0.0
cur_y=0.0
cur_z=0.0

echo "Initializing position to center..."
publish_cmd "$TOPIC_X" $cur_x
publish_cmd "$TOPIC_Y" $cur_y
publish_cmd "$TOPIC_Z" $cur_z
sleep 2

# direction_mode: 0=向外螺旋(Center->Edge), 1=向内螺旋(Edge->Center)
direction_mode=0 

for (( h=0; h<=HEIGHT; h+=STEP_SIZE )); do
    
    # 计算当前高度下的最大边长
    # 比例关系: current_max_side / BASE_SIZE = (HEIGHT - h) / HEIGHT
    max_side=$((((HEIGHT - h)* BASE_SIZE) / HEIGHT))
    
    echo "=== Layer Height: $h, Max Side: $max_side, Mode: $((direction_mode==0?Outward:Inward)) ==="

    # 如果已经到达顶点(边长0)，结束循环
    if [ "$max_side" -lt 0 ]; then
        break
    fi

    # 生成当前层的螺旋路径点 (仅 Y 和 Z 轴)
    # 临时数组清空
    path_y=()
    path_z=()
    
    # 起点中心
    sim_y=0
    sim_z=0
    path_y+=($sim_y)
    path_z+=($sim_z)

    # 螺旋生成算法 (Center -> Out)
    # 逻辑：右移k, 上移k, 左移k+step, 下移k+step...
    len=4
    # 方向标志: 1=Right/Up, -1=Left/Down
    sign=1 
    
    loop_flag=1
    while [ $loop_flag -eq 1 ]; do
        # 只要当前的螺旋臂长度还未超过最大边长，就继续生成
        if [ $len -gt $max_side ]; then
            loop_flag=0
            break
        fi

        # --- Y轴移动 (Right or Left) ---
        # 移动距离为 len，每次移动 STEP_SIZE (4)
        steps=$((len / STEP_SIZE))
        for (( i=0; i<steps; i++ )); do
            sim_y=$((sim_y + sign * STEP_SIZE))
            path_y+=($sim_y)
            path_z+=($sim_z) # Z保持不变
        done

        # --- Z轴移动 (Up or Down) ---
        steps=$((len / STEP_SIZE))
        for (( i=0; i<steps; i++ )); do
            sim_z=$((sim_z + sign * STEP_SIZE))
            path_y+=($sim_y) # Y保持不变
            path_z+=($sim_z)
        done

        # 准备下一次迭代
        sign=$((-1 * sign)) # 反转方向
        len=$((len + STEP_SIZE)) # 边长增加
    done

    # 根据模式执行路径
    path_len=${#path_y[@]}
    
    if [ "$direction_mode" -eq 0 ]; then
        # === 向外螺旋 (正序: 0 -> end) ===
        for (( i=0; i<path_len; i++ )); do
            target_y=${path_y[$i]}
            target_z=${path_z[$i]}
            
            # 只有当坐标发生变化时才发布（除了第一个点）
            # 或者为了确保目标到达，每一秒都发布当前目标
            echo "Moving to Y:$target_y Z:$target_z"
            publish_cmd "$TOPIC_Y" "$target_y"
            publish_cmd "$TOPIC_Z" "$target_z"
            sleep "$SLEEP_TIME"
        done
    else
        # === 向内螺旋 (倒序: end-1 -> 0) ===
        # 注意：上一层结束时，无人机在外部边界。
        # 我们需要从当前层对应的外部边界开始往里走。
        for (( i=path_len-1; i>=0; i-- )); do
            target_y=${path_y[$i]}
            target_z=${path_z[$i]}
            
            echo "Moving to Y:$target_y Z:$target_z"
            publish_cmd "$TOPIC_Y" "$target_y"
            publish_cmd "$TOPIC_Z" "$target_z"
            sleep "$SLEEP_TIME"
        done
    fi

    # 切换高度 (X轴)
    # 只有当还没有到达顶点时才移动高度
    if [ "$h" -lt "$HEIGHT" ]; then
        next_h=$((h + STEP_SIZE))
        #  -x_axis_joint 为高的方向，即向上运动为负值增加
        # 这里假设 0 是底面，-40 是顶点。
        # data = -next_h
        cmd_x=$(echo "-1 * $next_h" | bc)
        
        echo ">>> Stepping Up Height to $next_h (Command: $cmd_x) <<<"
        publish_cmd "$TOPIC_X" "$cmd_x"
        sleep "$SLEEP_TIME"
    fi

    # 切换下一层的螺旋模式
    if [ "$direction_mode" -eq 0 ]; then
        direction_mode=1 # 下一次向内
    else
        direction_mode=0 # 下一次向外
    fi

done

echo "Mission Complete. Reached Apex."