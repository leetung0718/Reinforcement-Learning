# AI for Logistics - Robots in a warehouse
# 物流AI - 倉庫中的機器人

import numpy as np

# 設定參數 gamma 和 alpha
GAMMA = 0.75  # 折扣因子，用於計算未來獎勵的現值
ALPHA = 0.9   # 學習速率，用於更新Q值

# 1. 建立環境
# 定義狀態
location_to_state = {'A': 0, 'B': 1, 'C': 2, 'D': 3,
                     'E': 4, 'F': 5, 'G': 6, 'H': 7,
                     'I': 8, 'J': 9, 'K': 10, 'L': 11}  # 地點到狀態的映射

state_to_location = {value: key for key, value in location_to_state.items()}  # 狀態到地點的映射

# 定義行動
actions = [0,1,2,3,4,5,6,7,8,9,10,11]  # 可能的行動，即移動到不同的狀態

# 定義獎勵
R = np.array([[0,1,0,0,0,0,0,0,0,0,0,0],
              [1,0,1,0,0,1,0,0,0,0,0,0],
              [0,1,0,0,0,0,1,0,0,0,0,0],
              [0,0,0,0,0,0,0,1,0,0,0,0],
              [0,0,0,0,0,0,0,0,1,0,0,0],
              [0,1,0,0,0,0,0,0,0,1,0,0],
              [0,0,1,0,0,0,1,1,0,0,0,0],
              [0,0,0,1,0,0,1,0,0,0,0,1],
              [0,0,0,0,1,0,0,0,0,1,0,0],
              [0,0,0,0,0,1,0,0,1,0,1,0],
              [0,0,0,0,0,0,0,0,0,1,0,1],
              [0,0,0,0,0,0,0,1,0,0,1,0]]) # 獎勵矩陣，表示從一個地點移動到另一個地點的獎勵值

# 2. 使用Q學習建立AI解決方案
# 實施Q學習過程
def route(starting_location, ending_location):
    R_new = np.copy(R)  # 複製獎勵矩陣
    ending_state = location_to_state[ending_location]  # 結束位置的狀態
    R_new[ending_state, ending_state] = 1000  # 設定目的地的獎勵為1000

    Q = np.array(np.zeros([12,12]))  # 初始化Q矩陣

    # Q學習演算法
    for _ in range(1000):  # 運行1000次以更新Q矩陣
        current_state = np.random.randint(0,12)  # 隨機選擇一個初始狀態
        playable_actions = []
        for j in range(12):  # 尋找可執行的行動
            if R_new[current_state, j] > 0:
                playable_actions.append(j)
        
        next_state = np.random.choice(playable_actions)  # 隨機選擇下一個狀態
        TD = R_new[current_state, next_state] + GAMMA * Q[next_state, np.argmax(Q[next_state,])] - Q[current_state, next_state]
        Q[current_state, next_state] = Q[current_state, next_state] + ALPHA * TD

    # 計算最佳路徑
    route = [starting_location]
    next_location = starting_location
    while (next_location != ending_location):
        starting_state = location_to_state[starting_location]
        next_state = np.argmax(Q[starting_state, ])
        next_location = state_to_location[next_state]
        route.append(next_location)
        starting_location = next_location
    
    return route
        
# 測試從E到K的路徑
print(route('E', 'K'))
