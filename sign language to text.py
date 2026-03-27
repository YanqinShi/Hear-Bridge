import mediapipe as mp
import cv2
import numpy as np


# ====== 口红函数 ======
def change_color_lip(img, list_lms, index_lip_up, index_lip_down, color):
    mask = np.zeros_like(img)

    points_lip_up = list_lms[index_lip_up, :]
    points_lip_down = list_lms[index_lip_down, :]

    cv2.fillPoly(mask, [points_lip_up], (255, 255, 255))
    cv2.fillPoly(mask, [points_lip_down], (255, 255, 255))

    img_color_lip = np.zeros_like(img)
    img_color_lip[:] = color

    img_color_lip = cv2.bitwise_and(mask, img_color_lip)

    # 模糊边缘，让口红更自然
    img_color_lip = cv2.GaussianBlur(img_color_lip, (7, 7), 10)

    # 融合
    result = cv2.addWeighted(img, 1, img_color_lip, 0.6, 0)

    return result


# ====== 手势识别函数 ======
def get_angle(v1, v2):
    angle = np.dot(v1, v2) / (np.sqrt(np.sum(v1 * v1)) * np.sqrt(np.sum(v2 * v2)))
    angle = np.arccos(angle) / 3.14 * 180
    return angle


def get_str_guester(up_fingers, list_lms):
    if len(up_fingers) == 1 and up_fingers[0] == 8:
        v1 = list_lms[6] - list_lms[7]
        v2 = list_lms[8] - list_lms[7]
        angle = get_angle(v1, v2)
        if angle < 160:
            str_guester = "9"
        else:
            str_guester = "1"
    elif len(up_fingers) == 1 and up_fingers[0] == 4:
        str_guester = "Good"
    elif len(up_fingers) == 1 and up_fingers[0] == 20:
        str_guester = "Bad"
    elif len(up_fingers) == 1 and up_fingers[0] == 12:
        str_guester = "FXXX"
    elif len(up_fingers) == 2 and up_fingers[0] == 8 and up_fingers[1] == 12:
        str_guester = "2"
    elif len(up_fingers) == 2 and up_fingers[0] == 4 and up_fingers[1] == 20:
        str_guester = "6"
    elif len(up_fingers) == 2 and up_fingers[0] == 4 and up_fingers[1] == 8:
        str_guester = "8"
    elif len(up_fingers) == 3 and up_fingers[0] == 8 and up_fingers[1] == 12 and up_fingers[2] == 16:
        str_guester = "3"
    elif len(up_fingers) == 3 and up_fingers[0] == 4 and up_fingers[1] == 8 and up_fingers[2] == 12:
        dis_8_12 = list_lms[8, :] - list_lms[12, :]
        dis_8_12 = np.sqrt(np.dot(dis_8_12, dis_8_12))
        dis_4_12 = list_lms[4, :] - list_lms[12, :]
        dis_4_12 = np.sqrt(np.dot(dis_4_12, dis_4_12))
        if dis_4_12 / (dis_8_12 + 1) < 3:
            str_guester = "7"
        elif dis_4_12 / (dis_8_12 + 1) > 5:
            str_guester = "Gun"
        else:
            str_guester = "7"
    elif len(up_fingers) == 3 and up_fingers[0] == 4 and up_fingers[1] == 8 and up_fingers[2] == 20:
        str_guester = "ROCK"
    elif len(up_fingers) == 4 and up_fingers[0] == 8 and up_fingers[1] == 12 and up_fingers[2] == 16 and up_fingers[
        3] == 20:
        str_guester = "4"
    elif len(up_fingers) == 5:
        str_guester = "5"
    elif len(up_fingers) == 0:
        str_guester = "10"
    else:
        str_guester = " "
    return str_guester


def empty(a):
    pass


# ====== 主程序 ======
if __name__ == "__main__":
    # 初始化人脸检测
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    mp_hand_drawing = mp.solutions.drawing_utils

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    hands = mp_hands.Hands(
        max_num_hands=4,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cv2.namedWindow("Face & Hand Detection")
    cv2.resizeWindow("Face & Hand Detection", 640, 240)
    cv2.createTrackbar("Blue", "Face & Hand Detection", 0, 255, empty)
    cv2.createTrackbar("Green", "Face & Hand Detection", 0, 255, empty)
    cv2.createTrackbar("Red", "Face & Hand Detection", 0, 255, empty)

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    while True:
        success, img = cap.read()
        if not success:
            continue

        image_height, image_width, _ = img.shape
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face_info = []
        hand_info = []

        # ===== 人脸检测 =====
        face_results = face_mesh.process(img_RGB)
        if face_results.multi_face_landmarks:
            for face_idx, face_landmarks in enumerate(face_results.multi_face_landmarks):
                mp_drawing.draw_landmarks(
                    img, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1)
                )
                # 获取坐标、画口红等（略，同之前）...
                # 收集信息
                face_info.append(f"Face{face_idx+1}")

        # ===== 手部检测 =====
        hand_results = hands.process(img_RGB)
        if hand_results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                # 绘制手部
                mp_hand_drawing.draw_landmarks(
                    img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_hand_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
                    mp_hand_drawing.DrawingSpec(color=(255,255,255), thickness=2)
                )

                # 采集坐标
                list_lms = []
                for i in range(21):
                    pos_x = hand_landmarks.landmark[i].x * image_width
                    pos_y = hand_landmarks.landmark[i].y * image_height
                    list_lms.append([int(pos_x), int(pos_y)])
                list_lms = np.array(list_lms, dtype=np.int32)

                # 凸包
                hull_index = [0,1,2,3,6,10,14,19,18,17,10]
                hull = cv2.convexHull(list_lms[hull_index, :])
                cv2.polylines(img, [hull], True, (0,255,0), 2)

                # 判断伸出的手指
                ll = [4,8,12,16,20]
                up_fingers = []
                for i in ll:
                    pt = (int(list_lms[i][0]), int(list_lms[i][1]))
                    dist = cv2.pointPolygonTest(hull, pt, True)
                    if dist < 0:
                        up_fingers.append(i)

                str_guester = get_str_guester(up_fingers, list_lms)

                # ===== 修正左右手标签 =====
                hand_label = "Unknown"
                if hand_results.multi_handedness:
                    original_label = hand_results.multi_handedness[hand_idx].classification[0].label
                    # 前置摄像头镜像，交换左右
                    if original_label == "Left":
                        hand_label = "Right"
                    elif original_label == "Right":
                        hand_label = "Left"
                    else:
                        hand_label = original_label

                # 显示文字
                wrist_pos = (int(hand_landmarks.landmark[0].x * image_width),
                             int(hand_landmarks.landmark[0].y * image_height))
                cv2.putText(img, f'{hand_label} Hand {hand_idx+1}: {str_guester}',
                            (wrist_pos[0]-50, wrist_pos[1]-50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

                # 标记指尖
                for i in ll:
                    pos_x = hand_landmarks.landmark[i].x * image_width
                    pos_y = hand_landmarks.landmark[i].y * image_height
                    cv2.circle(img, (int(pos_x), int(pos_y)), 5, (0,255,255), -1)

                # 收集手部信息
                index_tip = (int(hand_landmarks.landmark[8].x * image_width),
                             int(hand_landmarks.landmark[8].y * image_height))
                hand_info.append(f"Hand{hand_idx+1}({hand_label}): {len(up_fingers)} fingers, idx_tip({index_tip[0]},{index_tip[1]})")

        # 统计显示
        face_count = len(face_results.multi_face_landmarks) if face_results.multi_face_landmarks else 0
        hand_count = len(hand_results.multi_hand_landmarks) if hand_results.multi_hand_landmarks else 0
        cv2.putText(img, f'Faces: {face_count} | Hands: {hand_count}',
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        # 控制台输出
        output_str = f"Faces:{face_count} "
        if face_info:
            output_str += "[" + " | ".join(face_info) + "] "
        else:
            output_str += "[] "
        output_str += f"Hands:{hand_count} "
        if hand_info:
            output_str += "[" + " | ".join(hand_info) + "]"
        else:
            output_str += "[]"
        print(f"\r{output_str}", end="", flush=True)

        cv2.imshow("Face & Hand Detection", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()