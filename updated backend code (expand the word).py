import mediapipe as mp
import cv2
import numpy as np


# ====== 口红函数 ======
def change_color_lip(img, face_lms, index_lip_up, index_lip_down, color):
    mask = np.zeros_like(img)
    points_lip_up = face_lms[index_lip_up, :]
    points_lip_down = face_lms[index_lip_down, :]

    cv2.fillPoly(mask, [points_lip_up], (255, 255, 255))
    cv2.fillPoly(mask, [points_lip_down], (255, 255, 255))

    img_color_lip = np.zeros_like(img)
    img_color_lip[:] = color
    img_color_lip = cv2.bitwise_and(mask, img_color_lip)

    # 模糊边缘，让口红更自然
    img_color_lip = cv2.GaussianBlur(img_color_lip, (7, 7), 10)
    # 融合
    result = cv2.addWeighted(img, 1, img_color_lip, 0.4, 0)
    return result


def empty(a):
    pass


# ====== 词汇识别函数 ======
def get_sign_word(up_fingers, list_lms):
    word = ""
    fingers_tuple = tuple(sorted(up_fingers))

    if fingers_tuple == (12, 16, 20):
        dis_4_8 = np.linalg.norm(list_lms[4, :] - list_lms[8, :])
        if dis_4_8 < 30:
            word = "OK"
    elif fingers_tuple == (4, 8, 20):
        word = "Love"
    elif fingers_tuple == (4,):
        if list_lms[4][1] < list_lms[3][1]:
            word = "Good"
        else:
            word = "Bad"
    elif fingers_tuple == (8, 12):
        word = "Peace"
    elif fingers_tuple == (4, 8, 12, 16, 20):
        word = "Hello"
    elif fingers_tuple == (8,):
        word = "Look"

    return word


# ====== 主程序 ======
if __name__ == "__main__":
    # 1. 初始化人脸和手部模型
    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    # 2. UI 窗口与调色板
    cv2.namedWindow("Translator & Face")
    cv2.createTrackbar("Blue", "Translator & Face", 0, 255, empty)
    cv2.createTrackbar("Green", "Translator & Face", 0, 255, empty)
    cv2.createTrackbar("Red", "Translator & Face", 150, 255, empty)  # 默认给点红色

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    # 上下嘴唇的特征点索引（简化版）
    LIP_UP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78]
    LIP_DOWN = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61]

    # --- 翻译器状态变量 ---
    current_sentence = []
    prev_word = ""
    hold_frames = 0
    frames_threshold = 15
    cooldown_frames = 0

    while True:
        success, img = cap.read()
        if not success:
            continue

        # 水平翻转，符合直觉
        img = cv2.flip(img, 1)
        image_height, image_width, _ = img.shape
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 获取滑动条的颜色值
        b = cv2.getTrackbarPos("Blue", "Translator & Face")
        g = cv2.getTrackbarPos("Green", "Translator & Face")
        r = cv2.getTrackbarPos("Red", "Translator & Face")
        lip_color = (b, g, r)

        # ====== 处理人脸 ======
        face_results = face_mesh.process(img_RGB)
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                # 画人脸网格
                mp_drawing.draw_landmarks(
                    img, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                )

                # 提取面部坐标点用于口红
                face_lms = []
                for lm in face_landmarks.landmark:
                    face_lms.append([int(lm.x * image_width), int(lm.y * image_height)])
                face_lms = np.array(face_lms, dtype=np.int32)

                # 涂口红
                if r > 0 or g > 0 or b > 0:  # 只有当颜色不全为0时才涂色
                    img = change_color_lip(img, face_lms, LIP_UP, LIP_DOWN, lip_color)

        # ====== 处理手部与手语 ======
        hand_results = hands.process(img_RGB)
        detected_word = ""

        if hand_results.multi_hand_landmarks:
            hand_landmarks = hand_results.multi_hand_landmarks[0]

            # 画手部骨架
            mp_drawing.draw_landmarks(
                img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
            )

            # 采集坐标
            list_lms = []
            for i in range(21):
                pos_x = hand_landmarks.landmark[i].x * image_width
                pos_y = hand_landmarks.landmark[i].y * image_height
                list_lms.append([int(pos_x), int(pos_y)])
            list_lms = np.array(list_lms, dtype=np.int32)

            # 凸包判断伸出的手指
            hull_index = [0, 1, 2, 3, 6, 10, 14, 19, 18, 17, 10]
            hull = cv2.convexHull(list_lms[hull_index, :])
            cv2.polylines(img, [hull], True, (0, 255, 255), 2)

            ll = [4, 8, 12, 16, 20]
            up_fingers = []
            for i in ll:
                pt = (int(list_lms[i][0]), int(list_lms[i][1]))
                dist = cv2.pointPolygonTest(hull, pt, True)
                if dist < 0:
                    up_fingers.append(i)

            detected_word = get_sign_word(up_fingers, list_lms)

            # 手腕处显示当前状态
            if detected_word:
                wrist_pos = (list_lms[0][0], list_lms[0][1])
                cv2.putText(img, f'Current: {detected_word}',
                            (wrist_pos[0] - 50, wrist_pos[1] - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # ====== 翻译句子防抖逻辑 ======
        if cooldown_frames > 0:
            cooldown_frames -= 1
        else:
            if detected_word != "" and detected_word == prev_word:
                hold_frames += 1
                if hold_frames == frames_threshold:
                    current_sentence.append(detected_word)
                    cooldown_frames = 30
                    hold_frames = 0
            else:
                prev_word = detected_word
                hold_frames = 0

        # ====== 界面显示 ======
        sentence_str = " ".join(current_sentence)
        cv2.rectangle(img, (0, 0), (image_width, 80), (0, 0, 0), -1)
        cv2.putText(img, f'Translation: {sentence_str}', (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(img, "Press 'C' to clear | 'Q' to quit", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.imshow("Translator & Face", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            current_sentence = []

    cap.release()
    cv2.destroyAllWindows()