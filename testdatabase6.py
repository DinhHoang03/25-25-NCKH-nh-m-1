import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Danh sách toàn cục để lưu các mảng answer_key
all_answer_keys = []

# Hàm nhận diện mã đề thi hoặc số báo danh (không vẽ lên ảnh)
def get_code_fixed(image, is_exam_code=True):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.medianBlur(mask, 5)

    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1.2, minDist=15,
                               param1=50, param2=18, minRadius=12, maxRadius=25)

    detected_numbers = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for (x, y, r) in circles[0, :]:
            if 100 <= y <= 600:
                digit = (y - 100) // 50
                detected_numbers.append((x, digit))

    detected_numbers.sort(key=lambda x: x[0])
    code = "".join(str(num[1]) for num in detected_numbers)
    return code

# Hàm nhận diện các ô tròn được tô (không vẽ lên ảnh)
def detect_marked_circles(image, is_code=False, is_exam_code=True, column_index=None):
    translate = {0: "A", 1: "B", 2: "C", 3: "D", -1: "N"}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=50, param2=30, minRadius=12, maxRadius=35)

    selected_values = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        questionCnts = [(x, y, r) for (x, y, r) in circles]
        
        if len(questionCnts) == 0:
            print("❌ Không tìm thấy ô tròn!")
            return []

        questionCnts = sorted(questionCnts, key=lambda c: (c[1], c[0]))
        contours_list = [np.array([[[x - r, y - r]], [[x + r, y - r]], [[x + r, y + r]], [[x - r, y + r]]], dtype=np.int32)
                         for (x, y, r) in questionCnts]

        threshold_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                               cv2.THRESH_BINARY_INV, 15, 8)

        if is_code:
            code = get_code_fixed(image, is_exam_code)
            selected_values = list(code) if code else ["No code detected"]
            return selected_values
        else:
            for i in range(0, len(contours_list), 4):
                if i + 4 > len(contours_list):
                    break
                cnts = sorted(contours_list[i:i + 4], key=lambda c: cv2.boundingRect(c)[0])
                list_total = []
                total_max = -1

                for j, c in enumerate(cnts):
                    mask = np.zeros(threshold_image.shape, dtype="uint8")
                    cv2.drawContours(mask, [c], -1, 255, -1)
                    mask = cv2.bitwise_and(threshold_image, threshold_image, mask=mask)
                    total = cv2.countNonZero(mask)
                    total_max = max(total_max, total)
                    list_total.append((total, j))

                list_select = ''
                for tt in list_total:
                    if tt[0] > total_max * 0.7:
                        list_select += translate[tt[1]]

                if list_select:
                    question = (column_index * 30) + (i // 4) + 1
                    selected_values.append((question, list_select[0]))

            return selected_values

    return []

# Hàm chấm điểm bài thi (đã sửa)
def get_result_trac_nghiem(image_trac_nghiem, ANSWER_KEY):
    if image_trac_nghiem is None or image_trac_nghiem.size == 0:
        print("❌ Ảnh đầu vào rỗng!")
        return [], image_trac_nghiem

    translate = {"A": 0, "B": 1, "C": 2, "D": 3}
    revert_translate = {0: "A", 1: "B", 2: "C", 3: "D", -1: "N"}
    
    image = image_trac_nghiem.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=50, param2=30, minRadius=12, maxRadius=30)
    
    questionCnts = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            questionCnts.append((x, y, r))

    if len(questionCnts) == 0:
        print("❌ Không tìm thấy ô tròn!")
        return [], image

    questionCnts = sorted(questionCnts, key=lambda c: (c[1], c[0]))
    contours_list = [np.array([[[x - r, y - r]], [[x + r, y - r]], [[x + r, y + r]], [[x - r, y + r]]], dtype=np.int32)
                     for (x, y, r) in questionCnts]

    threshold_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                            cv2.THRESH_BINARY_INV, 15, 8)

    select = []
    for i in range(0, len(contours_list), 4):
        if i + 4 > len(contours_list):
            break
        cnts = sorted(contours_list[i:i + 4], key=lambda c: cv2.boundingRect(c)[0])
        list_total = []
        total_max = -1

        for j, c in enumerate(cnts):
            mask = np.zeros(threshold_image.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.bitwise_and(threshold_image, threshold_image, mask=mask)
            total = cv2.countNonZero(mask)
            total_max = max(total_max, total)
            list_total.append((total, j))

        answer_q = ANSWER_KEY[min(i // 4, len(ANSWER_KEY) - 1)]  # Đáp án chuẩn
        list_answer = []
        list_select = ''

        for tt in list_total:
            if tt[0] > total_max * 0.7:
                list_answer.append(tt[1])
                list_select += revert_translate[tt[1]]

        if not list_select:
            list_select = "N"

        correct_answer_idx = translate[answer_q]
        for j, c in enumerate(cnts):
            center_x = (c[0][0][0] + c[2][0][0]) // 2
            center_y = (c[0][0][1] + c[2][0][1]) // 2
            if j == correct_answer_idx and j in list_answer:
                cv2.circle(image, (center_x, center_y), 15, (0, 255, 0), 3)
            elif j in list_answer and j != correct_answer_idx:
                cv2.circle(image, (center_x, center_y), 15, (0, 0, 255), 3)

        select.append(list_select)

    return select, image

# Hàm tạo mảng answer_key từ SBD, mã đề thi và đáp án
def create_answer_key(sbd, exam_code, answers):
    answers.sort(key=lambda x: x[0])
    sorted_answers = [answer[1] for answer in answers]
    return [sbd, exam_code] + sorted_answers

# Biến toàn cục cho tọa độ crop
crop_answers = [
    (0, 790, 448, 2555),    # Câu 1-30
    (448, 790, 896, 2555),  # Câu 31-60
    (896, 790, 1344, 2555), # Câu 61-90
    (1344, 790, 1792, 2555) # Câu 91-120
]
# crop_answers = [
#         (0, 770, 430, 2555),   # Câu 1-30
#         (430, 770, 860, 2555),  # Câu 31-60
#         (860, 770, 1290, 2555), # Câu 61-90
#         (1290, 770, 1720, 2555) # Câu 91-120
# ]

# Hàm xử lý ảnh khi người dùng chọn file để lưu đáp án (không vẽ lên ảnh)
def process_image(file_path, window, result_label, image_label):
    original_image = cv2.imread(file_path)
    if original_image is None:
        messagebox.showerror("Lỗi", f"Không thể đọc ảnh: {file_path}")
        return

    crop_sbd = (1281, 154, 1550, 821)
    sbd_image = original_image[crop_sbd[1]:crop_sbd[3], crop_sbd[0]:crop_sbd[2]]
    sbd_data = detect_marked_circles(sbd_image, is_code=True, is_exam_code=False)
    sbd = ''.join(sbd_data) if sbd_data and sbd_data != ["No code detected"] else "No SBD detected"

    crop_exam_code = (1558, 154, 1726, 821)
    exam_code_image = original_image[crop_exam_code[1]:crop_exam_code[3], crop_exam_code[0]:crop_exam_code[2]]
    exam_code_data = detect_marked_circles(exam_code_image, is_code=True, is_exam_code=True)
    exam_code = ''.join(exam_code_data) if exam_code_data and exam_code_data != ["No code detected"] else "No code detected"

    all_answers = []
    for idx, crop_coords in enumerate(crop_answers):
        answers_image = original_image[crop_coords[1]:crop_coords[3], crop_coords[0]:crop_coords[2]]
        if answers_image is not None:
            answers_data = detect_marked_circles(answers_image, is_code=False, column_index=idx)
            all_answers.extend(answers_data)

    answer_key = create_answer_key(sbd, exam_code, all_answers)
    all_answer_keys.append(answer_key)

    result_text = f"SBD: {answer_key[0]}\nMã đề thi: {answer_key[1]}\nĐáp án: {', '.join(answer_key[2:])}"
    result_label.config(text=result_text)

    output_image = cv2.resize(original_image, (400, 400))
    output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(output_image_rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    image_label.config(image=imgtk)
    image_label.image = imgtk

    cv2.imwrite('output_with_results.jpg', original_image)
    messagebox.showinfo("Thành công", "Đã xử lý và lưu đáp án vào mảng!")

# Hàm mở file ảnh từ máy tính để lưu đáp án
def open_file(window, result_label, image_label):
    file_path = filedialog.askopenfilename(
        title="Chọn file ảnh để lưu đáp án",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if file_path:
        process_image(file_path, window, result_label, image_label)

# Hàm hiển thị các mảng answer_key đã lưu
def show_saved_arrays(result_label):
    if not all_answer_keys:
        result_label.config(text="Chưa có mảng nào được lưu!")
        return

    display_text = "Các mảng đã lưu:\n\n"
    for i, answer_key in enumerate(all_answer_keys, 1):
        display_text += f"Bài thi {i}:\n"
        display_text += f"SBD: {answer_key[0]}\n"
        display_text += f"Mã đề thi: {answer_key[1]}\n"
        display_text += f"Đáp án: {', '.join(answer_key[2:])}\n\n"
    
    result_label.config(text=display_text)

# Hàm chấm điểm dựa trên SBD trùng khớp
def grade_exam(window, result_label, image_label):
    if not all_answer_keys:
        messagebox.showwarning("Cảnh báo", "Chưa có đáp án nào trong mảng để chấm điểm!")
        return

    file_path = filedialog.askopenfilename(
        title="Chọn file ảnh để chấm điểm",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if not file_path:
        return

    original_image = cv2.imread(file_path)
    if original_image is None:
        messagebox.showerror("Lỗi", f"Không thể đọc ảnh: {file_path}")
        return

    crop_sbd = (1281, 154, 1550, 821)
    sbd_image = original_image[crop_sbd[1]:crop_sbd[3], crop_sbd[0]:crop_sbd[2]]
    sbd_data = detect_marked_circles(sbd_image, is_code=True, is_exam_code=False)
    sbd = ''.join(sbd_data) if sbd_data and sbd_data != ["No code detected"] else "No SBD detected"

    ANSWER_KEY = None
    for answer_key in all_answer_keys:
        if answer_key[0] == sbd:
            ANSWER_KEY = answer_key[2:]
            break

    if ANSWER_KEY is None:
        messagebox.showwarning("Cảnh báo", f"Không tìm thấy đáp án chuẩn cho SBD: {sbd} trong mảng!")
        return

    crop_exam_code = (1558, 154, 1726, 821)
    exam_code_image = original_image[crop_exam_code[1]:crop_exam_code[3], crop_exam_code[0]:crop_exam_code[2]]
    exam_code_data = detect_marked_circles(exam_code_image, is_code=True, is_exam_code=True)
    exam_code = ''.join(exam_code_data) if exam_code_data and exam_code_data != ["No code detected"] else "No code detected"

    all_answer_key = []
    for idx, (x1, y1, x2, y2) in enumerate(crop_answers):
        if y1 < 0 or y2 > original_image.shape[0] or x1 < 0 or x2 > original_image.shape[1]:
            print(f"❌ Tọa độ crop không hợp lệ cho vùng {idx}: ({x1}, {y1}, {x2}, {y2})")
            continue
        crop_img = original_image[y1:y2, x1:x2]
        if crop_img.size == 0:
            print(f"❌ Vùng crop {idx} rỗng!")
            continue
        ans, processed_img = get_result_trac_nghiem(crop_img, ANSWER_KEY[idx * 30: (idx + 1) * 30])
        all_answer_key.extend(ans)
        cv2.rectangle(original_image, (x1, y1), (x2, y2), (255, 0, 0), 3)
        original_image[y1:y2, x1:x2] = processed_img

    if len(all_answer_key) < len(ANSWER_KEY):
        all_answer_key.extend(["N"] * (len(ANSWER_KEY) - len(all_answer_key)))
    elif len(all_answer_key) > len(ANSWER_KEY):
        all_answer_key = all_answer_key[:len(ANSWER_KEY)]

    grading = [1 if str(ANSWER_KEY[x]) == str(all_answer_key[x]) else 0 for x in range(len(ANSWER_KEY))]
    score = (10 * sum(grading)) / len(ANSWER_KEY)
    score = round(score, 2)

    print(f"ANSWER_KEY (chuẩn): {ANSWER_KEY}")
    print(f"all_answer_key (bài chấm): {all_answer_key}")
    print(f"Số câu trong ANSWER_KEY: {len(ANSWER_KEY)}")
    print(f"Số câu trong all_answer_key: {len(all_answer_key)}")
    print(f"Grading (1: đúng, 0: sai): {grading}")
    print(f"Tổng số câu đúng: {sum(grading)}")
    print(f"Điểm: {score}")

    cv2.putText(original_image, f"Score: {score}/10", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    cv2.putText(original_image, f"SBD: {sbd}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    cv2.putText(original_image, f"Exam Code: {exam_code}", (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    output_image = cv2.resize(original_image, (400, 400))
    output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(output_image_rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    image_label.config(image=imgtk)
    image_label.image = imgtk

    result_text = f"SBD: {sbd}\nMã đề thi: {exam_code}\nĐiểm: {score}/10"
    result_label.config(text=result_text)

    cv2.imwrite('output_graded.jpg', original_image)
    messagebox.showinfo("Thành công", "Đã chấm điểm và lưu ảnh kết quả!")

# Hàm tạo giao diện Tkinter
def create_gui():
    window = tk.Tk()
    window.title("Nhận diện và chấm điểm bài thi trắc nghiệm")
    window.geometry("900x900")

    title_label = tk.Label(window, text="Chọn ảnh bài thi để nhận diện và chấm điểm", font=("Arial", 14))
    title_label.pack(pady=10)

    select_button = tk.Button(window, text="Chọn ảnh để lưu đáp án", command=lambda: open_file(window, result_label, image_label), font=("Arial", 12))
    select_button.pack(pady=10)

    show_arrays_button = tk.Button(window, text="Hiển thị mảng đã lưu", command=lambda: show_saved_arrays(result_label), font=("Arial", 12))
    show_arrays_button.pack(pady=10)

    grade_button = tk.Button(window, text="Chấm điểm", command=lambda: grade_exam(window, result_label, image_label), font=("Arial", 12))
    grade_button.pack(pady=10)

    result_label = tk.Label(window, text="Kết quả sẽ hiển thị ở đây", font=("Arial", 10), wraplength=500, justify="left")
    result_label.pack(pady=10)

    image_label = tk.Label(window)
    image_label.pack(pady=10)

    window.mainloop()

if __name__ == "__main__":
    create_gui()