import mediapipe as mp
import cv2
import numpy as np
import os
import json
from collections import deque
from PIL import Image, ImageDraw, ImageFont


# ====== 1. 字体引擎 ======
def load_fonts():
    font_paths = ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf", "/System/Library/Fonts/PingFang.ttc"]
    selected_path = next((p for p in font_paths if os.path.exists(p)), "arial.ttf")
    return {
        'title': ImageFont.truetype(selected_path, 24),
        'chat_new': ImageFont.truetype(selected_path, 36),
        'chat_old': ImageFont.truetype(selected_path, 22),
        'info_b': ImageFont.truetype(selected_path, 20),
        'info': ImageFont.truetype(selected_path, 16),
        'tip': ImageFont.truetype(selected_path, 14),
        'cv2_font': ImageFont.truetype(selected_path, 24),
        'cv2_big': ImageFont.truetype(selected_path, 40)
    }


def cv2_put_text_chinese(img, text, position, text_color, font):
    cv2_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im)
    draw = ImageDraw.Draw(pil_im)
    draw.text(position, text, font=font, fill=(text_color[2], text_color[1], text_color[0]))
    return cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)


# ====== 2. 双字典 & 记忆系统 (双手全能版) ======
REAL_CSL_DICT = {
    # 右手触发
    ("", "", "OpenHand", "Swipe Right"): "你好！",
    ("", "", "Good", "Swipe Down"): "谢谢！",
    ("", "", "OpenHand", "Swipe Left"): "不要 / 不是的",
    ("", "", "Fist", "Swipe Down"): "对不起 / 抱歉",
    ("", "", "OK", "Swipe Down"): "好的，没问题。",
    # 左手触发 (完全对称)
    ("OpenHand", "Swipe Right", "", ""): "你好！",
    ("Good", "Swipe Down", "", ""): "谢谢！",
    ("OpenHand", "Swipe Left", "", ""): "不要 / 不是的",
    ("Fist", "Swipe Down", "", ""): "对不起 / 抱歉",
    ("OK", "Swipe Down", "", ""): "好的，没问题。"
}

# 撤销与清空：现在左手和右手捏 OK 都能触发
SYS_DICT = {
    ("OK", "Swipe Left", "", ""): "Undo", ("OK", "Swipe Right", "", ""): "Clear",
    ("", "", "OK", "Swipe Left"): "Undo", ("", "", "OK", "Swipe Right"): "Clear"
}

CUSTOM_DICT_FILE = "custom_signs.json"


def load_custom_dict():
    if os.path.exists(CUSTOM_DICT_FILE):
        with open(CUSTOM_DICT_FILE, 'r', encoding='utf-8') as f:
            return {tuple(eval(k)): v for k, v in json.load(f).items()}
    return {}


def save_custom_dict(custom_dict):
    with open(CUSTOM_DICT_FILE, 'w', encoding='utf-8') as f:
        json.dump({str(k): v for k, v in custom_dict.items()}, f, ensure_ascii=False, indent=4)


# ====== 3. 高精度骨骼解算引擎 ======
def get_high_precision_gesture(hand_landmarks, image_width, image_height):
    lms = np.array([[lm.x * image_width, lm.y * image_height] for lm in hand_landmarks.landmark])
    up_fingers = []

    dist_4_17 = np.linalg.norm(lms[4] - lms[17])
    dist_3_17 = np.linalg.norm(lms[3] - lms[17])
    if dist_4_17 > dist_3_17: up_fingers.append(4)

    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        dist_tip = np.linalg.norm(lms[tip] - lms[0])
        dist_pip = np.linalg.norm(lms[pip] - lms[0])
        if dist_tip > dist_pip * 1.15: up_fingers.append(tip)

    word = ""
    fingers_tuple = tuple(sorted(up_fingers))
    palm_size = np.linalg.norm(lms[0] - lms[5])

    if len(fingers_tuple) == 0:
        word = "Fist"
    elif fingers_tuple == (8,):
        word = "One"
    elif fingers_tuple == (8, 12):
        word = "Two"
    elif fingers_tuple == (8, 12, 16):
        word = "Three"
    elif fingers_tuple == (8, 12, 16, 20):
        word = "Four"
    elif fingers_tuple == (4, 8, 12, 16, 20):
        word = "OpenHand"
    elif fingers_tuple == (4, 20):
        word = "Six"
    elif fingers_tuple == (12, 16, 20):
        if np.linalg.norm(lms[4] - lms[8]) < palm_size * 0.4: word = "OK"
    elif fingers_tuple == (4, 8, 20):
        word = "Love"
    elif fingers_tuple == (4,):
        word = "Good" if lms[4][1] < lms[3][1] else "Bad"
    elif fingers_tuple == (8, 20):
        word = "Rock"

    return word, lms[0]


# ====== 4. 动态 UI 引擎 ======
def draw_modern_dashboard(img_bg, current_sentences, state, fonts, mode, is_custom_active):
    img_pil = Image.fromarray(cv2.cvtColor(img_bg, cv2.COLOR_BGR2RGB)).convert('RGBA')
    ui_layer = Image.new('RGBA', img_pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(ui_layer)
    width, height = img_pil.size

    # [左侧]
    draw.rounded_rectangle([(20, 20), (450, height - 20)], radius=15, fill=(15, 20, 25, 200))
    title_text = "[ 高精度全能窗口 ]" if mode == "NORMAL" else "[ 录制模式中... ]"
    title_color = (255, 215, 0, 255) if mode == "NORMAL" else (255, 80, 80, 255)
    draw.text((40, 40), title_text, font=fonts['title'], fill=title_color)
    draw.line([(40, 80), (430, 80)], fill=(100, 110, 120, 100), width=1)

    if mode == "RECORDING":
        draw.text((40, 120), "请做动作，完成后再按 R 结束", font=fonts['chat_new'], fill=(255, 150, 150, 255))
    elif not current_sentences:
        draw.text((40, 120), "引擎就绪，左右手均可操作", font=fonts['chat_old'], fill=(100, 110, 120, 255))
    else:
        start_y = height - 120
        for i, sentence in enumerate(reversed(current_sentences)):
            if i == 0:
                draw.text((40, start_y), f"▶ {sentence}", font=fonts['chat_new'], fill=(50, 255, 180, 255))
            else:
                draw.text((40, start_y - i * 60 + 10), sentence, font=fonts['chat_old'], fill=(150, 160, 170, 255))

    # [右侧]
    panel_w = 280
    right_x = width - panel_w - 20
    draw.rounded_rectangle([(right_x, 20), (width - 20, 240)], radius=15, fill=(15, 20, 25, 200))
    draw.text((right_x + 20, 35), "[ 传感器状态 ]", font=fonts['info_b'], fill=(255, 255, 255, 255))

    for idx, (hand_name, label) in enumerate([("Left", "左手"), ("Right", "右手")]):
        stat = state[hand_name]["static"] or "--"
        act = state[hand_name]["action"] or "静止"
        color = (255, 160, 50, 255) if hand_name == "Left" else (50, 200, 255, 255)
        draw.text((right_x + 20, 80 + idx * 45), f"{label}:", font=fonts['info'], fill=(150, 150, 150, 255))
        draw.text((right_x + 70, 80 + idx * 45), f"[{stat}] {act}", font=fonts['info_b'], fill=color)

    dict_mode_text = "专属菜单字典 (右区)" if is_custom_active else "通用标准手语 (大区)"
    dict_mode_color = (0, 200, 255, 255) if is_custom_active else (150, 255, 150, 255)
    draw.line([(right_x + 20, 185), (width - 40, 185)], fill=(100, 110, 120, 100), width=1)
    draw.text((right_x + 20, 200), f"引擎: {dict_mode_text}", font=fonts['info'], fill=dict_mode_color)

    # 快捷键
    draw.rounded_rectangle([(right_x, height - 120), (width - 20, height - 20)], radius=10, fill=(15, 20, 25, 200))
    draw.text((right_x + 20, height - 105), "系统快捷键:", font=fonts['info'], fill=(200, 200, 200, 255))
    draw.text((right_x + 20, height - 80), "按 [R] : 开始/结束录制", font=fonts['tip'], fill=(255, 150, 150, 255))
    draw.text((right_x + 20, height - 55), "捏OK + 左滑 : 撤销动作", font=fonts['tip'], fill=(100, 150, 255, 255))

    out_img = Image.alpha_composite(img_pil, ui_layer)
    return cv2.cvtColor(np.array(out_img), cv2.COLOR_RGBA2BGR)


# ====== 5. 主循环 ======
if __name__ == "__main__":
    UI_FONTS = load_fonts()
    CUSTOM_DICT = load_custom_dict()

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.85, min_tracking_confidence=0.85)

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    histories = {"Left": deque(maxlen=15), "Right": deque(maxlen=15)}
    cooldowns = {"Left": 0, "Right": 0}
    current_sentences = []

    app_mode = "NORMAL"
    recorded_action_key = None

    # 专属词汇区占据右半屏
    CUSTOM_ZONE = (640, 80, 1260, 680)

    print("[系统] 双手全能引擎启动成功！")

    while True:
        success, img = cap.read()
        if not success: continue

        img = cv2.flip(img, 1)
        image_height, image_width, _ = img.shape
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        cv2.rectangle(img, (CUSTOM_ZONE[0], CUSTOM_ZONE[1]), (CUSTOM_ZONE[2], CUSTOM_ZONE[3]), (0, 165, 255), 2)
        img = cv2_put_text_chinese(img, "- 专属词汇区 (右半屏) -", (CUSTOM_ZONE[0] + 150, CUSTOM_ZONE[1] - 40),
                                   (0, 165, 255), UI_FONTS['cv2_font'])

        hand_results = hands.process(img_RGB)
        state = {"Left": {"static": "", "action": ""}, "Right": {"static": "", "action": ""}}

        # 记录每只手是否在专属区内
        zone_status = {"Left": False, "Right": False}

        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                hand_label = handedness.classification[0].label

                static_gesture, wrist_pos = get_high_precision_gesture(hand_landmarks, image_width, image_height)

                # 无论左手还是右手，只要在框内就标记为 True
                in_zone = (CUSTOM_ZONE[0] <= wrist_pos[0] <= CUSTOM_ZONE[2]) and (
                            CUSTOM_ZONE[1] <= wrist_pos[1] <= CUSTOM_ZONE[3])
                zone_status[hand_label] = in_zone

                # 视觉反馈：在框内统一为橙色圈，框外维持各自的默认颜色
                if in_zone:
                    cv2.circle(img, (int(wrist_pos[0]), int(wrist_pos[1])), 8, (0, 165, 255), 3)
                else:
                    color = (255, 160, 50) if hand_label == "Left" else (50, 200, 255)
                    cv2.circle(img, (int(wrist_pos[0]), int(wrist_pos[1])), 8, color, 2)

                histories[hand_label].append(wrist_pos)
                state[hand_label]["static"] = static_gesture

                if len(histories[hand_label]) == 15 and cooldowns[hand_label] == 0:
                    dx = histories[hand_label][-1][0] - histories[hand_label][0][0]
                    dy = histories[hand_label][-1][1] - histories[hand_label][0][1]

                    move_threshold_x = image_width * 0.10
                    move_threshold_y = image_height * 0.15

                    if dy > move_threshold_y and abs(dx) < move_threshold_x:
                        state[hand_label]["action"] = "Swipe Down"
                    elif dy < -move_threshold_y and abs(dx) < move_threshold_x:
                        state[hand_label]["action"] = "Swipe Up"
                    elif dx > move_threshold_x and abs(dy) < move_threshold_y:
                        state[hand_label]["action"] = "Swipe Right"
                    elif dx < -move_threshold_x and abs(dy) < move_threshold_y:
                        state[hand_label]["action"] = "Swipe Left"

        L_s, L_a = state["Left"]["static"], state["Left"]["action"]
        R_s, R_a = state["Right"]["static"], state["Right"]["action"]
        current_action_key = (L_s, L_a, R_s, R_a)

        # 智能判定：如果有动作发生，且发生动作的那只手在专属区，则激活专属字典
        is_custom_active = False
        if (L_a and zone_status["Left"]) or (R_a and zone_status["Right"]):
            is_custom_active = True

        # ==================================================
        # 模式控制与查字典
        # ==================================================
        sentence_to_add = ""

        if app_mode == "RECORDING" and (L_a or R_a):
            recorded_action_key = current_action_key

        elif app_mode == "NORMAL":
            sys_key = (L_s, L_a, "", "") if L_a else ("", "", R_s, R_a)  # 支持左右手系统指令

            if sys_key in SYS_DICT:
                if SYS_DICT[sys_key] == "Undo" and current_sentences:
                    current_sentences.pop()
                elif SYS_DICT[sys_key] == "Clear":
                    current_sentences.clear()
                cooldowns["Left"] = 25;
                cooldowns["Right"] = 25
                histories["Left"].clear();
                histories["Right"].clear()

            elif L_a or R_a:
                search_keys = [current_action_key, (L_s, "", R_s, R_a), ("", "", R_s, R_a), (L_s, L_a, "", "")]
                TARGET_DICT = CUSTOM_DICT if is_custom_active else REAL_CSL_DICT

                for key in search_keys:
                    if key in TARGET_DICT:
                        sentence_to_add = TARGET_DICT[key]
                        break

        # 更新队列
        if sentence_to_add:
            cooldowns["Left"], cooldowns["Right"] = 25, 25
            histories["Left"].clear();
            histories["Right"].clear()
            if len(current_sentences) >= 8: current_sentences.pop(0)
            current_sentences.append(sentence_to_add)

        if cooldowns["Left"] > 0: cooldowns["Left"] -= 1
        if cooldowns["Right"] > 0: cooldowns["Right"] -= 1

        img = draw_modern_dashboard(img, current_sentences, state, UI_FONTS, app_mode, is_custom_active)

        # ==================================================
        # 键盘事件监听 (R键起止)
        # ==================================================
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r') or key == ord('R'):
            if app_mode == "NORMAL":
                app_mode = "RECORDING"
                recorded_action_key = None
                current_sentences.append("[系统] 已开始录制，请做动作...")
                print("\n[系统] 已进入录制模式！请在摄像头前做出动作。完成后再次按 'R' 键。")

            elif app_mode == "RECORDING":
                if recorded_action_key:
                    app_mode = "PAUSED_FOR_INPUT"
                else:
                    app_mode = "NORMAL"
                    current_sentences.pop()
                    print("[系统] 未检测到滑动动作，已取消录制。")

        # ==================================================
        # 录制结束：输入环节
        # ==================================================
        if app_mode == "PAUSED_FOR_INPUT":
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (image_width, image_height), (0, 0, 0), 180)
            img = cv2.addWeighted(overlay, 0.8, img, 0.2, 0)
            img = cv2_put_text_chinese(img, "[ 动作已捕捉！请在后台终端输入文字 ]", (180, image_height // 2 - 40),
                                       (0, 255, 255), UI_FONTS['cv2_big'])
            cv2.imshow("Smart POS Assistant", img)
            cv2.waitKey(1)

            print("\n" + "=" * 50)
            print(">>> 动作捕捉成功！")
            print(f">>> 特征码: {recorded_action_key}")
            meaning = input(">>> 请输入绑定的词汇 (直接回车取消): ")

            if meaning.strip():
                CUSTOM_DICT[recorded_action_key] = meaning
                save_custom_dict(CUSTOM_DICT)
                print(f">>> 成功绑定: '{meaning}'")
                current_sentences.append(f"[系统] 新词汇已激活: {meaning}")
            else:
                print(">>> 已取消录入。")
                current_sentences.pop()
            print("=" * 50 + "\n")

            app_mode = "NORMAL"
            cooldowns["Left"] = 30;
            cooldowns["Right"] = 30
            histories["Left"].clear();
            histories["Right"].clear()
            continue

        cv2.imshow("Smart POS Assistant", img)

    cap.release()
    cv2.destroyAllWindows()