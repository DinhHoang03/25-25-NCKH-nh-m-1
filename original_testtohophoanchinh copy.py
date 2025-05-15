import cv2
import numpy as np
import os
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

def get_result_trac_nghiem(image_trac_nghiem, ANSWER_KEY):
    translate = {"A": 0, "B": 1, "C": 2, "D": 3}
    revert_translate = {0: "A", 1: "B", 2: "C", 3: "D", -1: "N"}
    
    image = image_trac_nghiem.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=50, param2=29, minRadius=12, maxRadius=30)
    
    questionCnts = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            questionCnts.append((x, y, r))
    
    if len(questionCnts) == 0:
        print("❌ Không tìm thấy đủ ô tròn! Hãy kiểm tra ảnh đầu vào.")
        return [], image, gray

    questionCnts = sorted(questionCnts, key=lambda c: (c[1], c[0]))

    contours_list = [np.array([[[x - r, y - r]], [[x + r, y - r]], [[x + r, y + r]], [[x - r, y + r]]], dtype=np.int32)
                     for (x, y, r) in questionCnts]

    threshold_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                            cv2.THRESH_BINARY_INV, 15, 8)
    select = []
    list_min_black = []
    min_black = float("inf")
    
    for i in range(0, len(contours_list), 4):
        if i + 4 > len(contours_list):
            break
        cnts = sorted(contours_list[i:i + 4], key=lambda c: cv2.boundingRect(c)[0])
        for c in cnts:
            mask = np.zeros(threshold_image.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.bitwise_and(threshold_image, threshold_image, mask=mask)
            total = cv2.countNonZero(mask)
            min_black = min(min_black, total)
        if (i + 4) % 20 == 0:
            list_min_black.append(min_black)
            min_black = float("inf")
    
    if not list_min_black:
        list_min_black = [100] * (len(contours_list) // 20 + 1)
    
    for i in range(0, len(contours_list), 4):
        if i + 4 > len(contours_list):
            break
        min_black = list_min_black[min(i // 20, len(list_min_black) - 1)]
        cnts = sorted(contours_list[i:i + 4], key=lambda c: cv2.boundingRect(c)[0])
        list_total = []
        total_max = -1
        
        for j, c in enumerate(cnts):
            mask = np.zeros(threshold_image.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.bitwise_and(threshold_image, threshold_image, mask=mask)
            total = cv2.countNonZero(mask)
            total_max = max(total_max, total)
            if total > 0:
                list_total.append((total, j))
        
        answer_q = [char for char in ANSWER_KEY[min(i // 4, len(ANSWER_KEY) - 1)]]
        list_answer = []
        list_select = ''
        
        for tt in list_total:
            if tt[0] > min_black * 1.5 and tt[0] > total_max * 0.7:
                list_answer.append(tt[1])
                list_select += revert_translate[tt[1]]
        
        for answer in answer_q:
            k = translate[answer]
            cv2.drawContours(image, [cnts[k]], -1, (0, 255, 0), 3)  # Đáp án đúng luôn tô xanh
        
        for j, c in enumerate(cnts):
            if j not in list_answer:
                cv2.drawContours(image, [c], -1, (0, 0, 255), 2)  # Tô đỏ các ô không được tô
        
        select.append(list_select)
    
    return select, image



def get_sbd_blue(image):
    # Chuyển ảnh sang HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Xác định phạm vi màu xanh dương trong không gian HSV
    lower_blue = np.array([90, 50, 50])   # Giới hạn dưới của màu xanh
    upper_blue = np.array([130, 255, 255]) # Giới hạn trên của màu xanh

    # Tạo mặt nạ chỉ giữ lại các vùng có màu xanh dương
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Dùng HoughCircles để tìm hình tròn đã tô màu xanh
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=50, param2=15, minRadius=6, maxRadius=20)

    detected_numbers = []

    if circles is not None:
        circles = np.uint16(np.around(circles))  # Làm tròn giá trị
        for (x, y, r) in circles[0, :]:
            y = int(y)  # Chuyển sang kiểu `int`
            
            if y >= 100:  # Kiểm tra y có hợp lệ không
                digit = (y - 100) // 50
            else:
                digit = 0  # Giá trị mặc định nếu y không hợp lệ
            
            detected_numbers.append((x, digit))

            # Vẽ vòng tròn màu đỏ để kiểm tra nhận diện
            cv2.circle(image, (x, y), r, (0, 0, 255), 2)
            cv2.putText(image, str(digit), (x - 10, y + 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Sắp xếp số báo danh theo vị trí từ trái sang phải
    detected_numbers.sort(key=lambda x: x[0])
    sbd = "".join(str(num[1]) for num in detected_numbers)

    return sbd, image

def get_mdt_blue(image):
    # Đọc ảnh

    # Chuyển ảnh sang không gian màu HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Xác định phạm vi màu xanh dương trong không gian HSV
    lower_blue = np.array([90, 50, 50])   # Giới hạn dưới của màu xanh
    upper_blue = np.array([130, 255, 255]) # Giới hạn trên của màu xanh

    # Tạo mặt nạ chỉ giữ lại các vùng có màu xanh dương
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Dùng HoughCircles để tìm hình tròn đã tô màu xanh
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=50, param2=15, minRadius=8, maxRadius=45)

    detected_numbers = []

    if circles is not None:
        circles = np.uint16(np.around(circles))  # Làm tròn giá trị
        for (x, y, r) in circles[0, :]:
            y = int(y)  # Chuyển sang kiểu `int`
            
            if y >= 100:  # Kiểm tra y có hợp lệ không
                digit = (y - 100) // 50  # Xác định số theo hàng
            else:
                digit = 0  # Giá trị mặc định nếu y không hợp lệ
            
            detected_numbers.append((x, digit))

            # Vẽ vòng tròn màu đỏ để kiểm tra nhận diện
            cv2.circle(image, (x, y), r, (0, 0, 255), 2)
            cv2.putText(image, str(digit), (x - 10, y + 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Sắp xếp mã đề theo vị trí từ trái sang phải
    detected_numbers.sort(key=lambda x: x[0])
    mdt = "".join(str(num[1]) for num in detected_numbers)

    return mdt, image

# --- Giao diện mới ---
class TracNghiemGraderGUI:
    def __init__(self):
        self.window = ttk.Window(title="Chấm Thi Trắc Nghiệm", themename="cosmo", size=(1280, 800), resizable=(True, True))
        self.window.place_window_center()
        self.answer_image_path = None
        self.answer_img_cv = None
        self.answer_key = None
        self.answer_exam_code = None
        self.image_path = None
        self.img_cv = None
        self.img_result = None
        self.result_info = None
        self._create_widgets()

    def _create_widgets(self):
        title = ttk.Label(self.window, text="📝 PHẦN MỀM CHẤM THI TRẮC NGHIỆM", font=("Segoe UI", 22, "bold"), bootstyle=PRIMARY)
        title.pack(pady=(18, 0))

        main_frame = ttk.Frame(self.window, padding=20)
        main_frame.pack(fill=BOTH, expand=YES)

        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=LEFT, fill=Y, padx=(0, 18))

        # Panel chức năng
        control_panel = ttk.LabelFrame(left_frame, text="CHỨC NĂNG", padding=18, bootstyle=PRIMARY)
        control_panel.pack(fill=X, pady=(0, 18))

        ttk.Button(control_panel, text="📑 Quét ảnh chấm thi", command=self._choose_answer_image, bootstyle=(INFO, OUTLINE), width=28).pack(pady=10)
        ttk.Button(control_panel, text="📂 Chọn ảnh chấm điểm", command=self._choose_image, bootstyle=(WARNING, OUTLINE), width=28).pack(pady=10)
        ttk.Button(control_panel, text="✔️ Chấm điểm", command=self._grade, bootstyle=(SUCCESS, OUTLINE), width=28).pack(pady=10)
        ttk.Button(control_panel, text="💾 Lưu ảnh chấm thi", command=self._save_result_image, bootstyle=(SECONDARY, OUTLINE), width=28).pack(pady=10)

        # Kết quả
        result_frame = ttk.LabelFrame(left_frame, text="KẾT QUẢ", padding=18, bootstyle=INFO)
        result_frame.pack(fill=BOTH, expand=YES)
        self.result_text = ttk.Text(result_frame, wrap=WORD, height=18, font=("Segoe UI", 12))
        self.result_text.pack(fill=BOTH, expand=YES)

        # Cột phải: preview ảnh
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=LEFT, fill=BOTH, expand=YES)

        preview_label = ttk.Label(right_frame, text="👁️ XEM TRƯỚC ẢNH NHẬN DIỆN", font=("Segoe UI", 13, "bold"), bootstyle=INFO)
        preview_label.pack(pady=(0, 8))

        preview_border = ttk.Frame(right_frame, borderwidth=3, relief="ridge", bootstyle=LIGHT)
        preview_border.pack(fill=BOTH, expand=YES, padx=10, pady=10)
        self.image_label = ttk.Label(preview_border, background="#f8f9fa")
        self.image_label.pack(fill=BOTH, expand=YES)

        self.status_bar = ttk.Label(self.window, text="Sẵn sàng", relief=SUNKEN, padding=7, anchor=W, font=("Segoe UI", 10))
        self.status_bar.pack(side=BOTTOM, fill=X)

    def _choose_answer_image(self):
        file_path = filedialog.askopenfilename(title="Chọn ảnh đáp án mẫu", filetypes=[("Ảnh", "*.jpg *.jpeg *.png")])
        if not file_path:
            return
        self.answer_image_path = file_path
        self.answer_img_cv = cv2.imread(file_path)
        img = self.answer_img_cv.copy()
        crop_regions = [
            (0, 790, 448, 2555),
            (448, 790, 896, 2555),
            (896, 790, 1344, 2555),
            (1344, 790, 1792, 2555)
        ]
        crop_mdt = (1558, 154, 1726, 821)
        crop_img_mdt = img[crop_mdt[1]:crop_mdt[3], crop_mdt[0]:crop_mdt[2]]
        mdt, _ = get_mdt_blue(crop_img_mdt)
        ANSWER_KEY = ["A", "B", "C", "D"] * 30
        all_answer_key = []
        for idx, (x1, y1, x2, y2) in enumerate(crop_regions):
            crop_img = img[y1:y2, x1:x2]
            ans, _ = get_result_trac_nghiem(crop_img, ANSWER_KEY[idx * 30: (idx + 1) * 30])
            all_answer_key.extend(ans)
        if len(all_answer_key) < len(ANSWER_KEY):
            all_answer_key.extend(["N"] * (len(ANSWER_KEY) - len(all_answer_key)))
        self.answer_key = all_answer_key
        self.answer_exam_code = mdt
        self.status_bar.config(text=f"Đã quét đáp án mẫu, mã đề: {mdt}")
        messagebox.showinfo("Thành Công", f"Đã nhận diện đáp án mẫu với mã đề: {mdt}\nSẵn sàng chấm phiếu học sinh!")
        self._show_image(self.answer_img_cv)

    def _choose_image(self):
        file_path = filedialog.askopenfilename(title="Chọn ảnh phiếu chấm điểm", filetypes=[("Ảnh", "*.jpg *.jpeg *.png")])
        if not file_path:
            return
        self.image_path = file_path
        self.img_cv = cv2.imread(file_path)
        self.img_result = None
        self.result_info = None
        self._show_image(self.img_cv)
        self.result_text.delete(1.0, 'end')
        self.status_bar.config(text="Đã chọn ảnh phiếu: " + os.path.basename(file_path))

    def _grade(self):
        if self.img_cv is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn ảnh phiếu trước!")
            return
        if self.answer_key is None or self.answer_exam_code is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng quét ảnh chấm thi trước!")
            return
        self.status_bar.config(text="Đang nhận diện và chấm điểm...")
        img = self.img_cv.copy()
        crop_regions = [
            (0, 790, 448, 2555),
            (448, 790, 896, 2555),
            (896, 790, 1344, 2555),
            (1344, 790, 1792, 2555)
        ]
        crop_sbd = (1281, 154, 1550, 821)
        crop_img_sbd = img[crop_sbd[1]:crop_sbd[3], crop_sbd[0]:crop_sbd[2]]
        sbd, _ = get_sbd_blue(crop_img_sbd)
        crop_mdt = (1558, 154, 1726, 821)
        crop_img_mdt = img[crop_mdt[1]:crop_mdt[3], crop_mdt[0]:crop_mdt[2]]
        mdt, _ = get_mdt_blue(crop_img_mdt)
        if mdt != self.answer_exam_code:
            self.result_text.delete(1.0, 'end')
            self.result_text.insert('end', f"❌ Mã đề thi không trùng với đáp án mẫu!\nMã đề phiếu: {mdt}\nMã đề đáp án mẫu: {self.answer_exam_code}\nKhông thể chấm điểm.")
            self.status_bar.config(text="Sai mã đề thi, không chấm điểm!")
            self._show_image(img)
            return
        all_answer_key = []
        preview_img = img.copy()
        for idx, (x1, y1, x2, y2) in enumerate(crop_regions):
            crop_img = img[y1:y2, x1:x2]
            ans, processed_img = get_result_trac_nghiem(crop_img, self.answer_key[idx * 30: (idx + 1) * 30])
            all_answer_key.extend(ans)
            cv2.rectangle(preview_img, (x1, y1), (x2, y2), (255, 0, 0), 3)
            try:
                preview_img[y1:y2, x1:x2] = processed_img
            except Exception:
                pass
        if len(all_answer_key) < len(self.answer_key):
            all_answer_key.extend(["N"] * (len(self.answer_key) - len(all_answer_key)))
        grading = [1 if str(self.answer_key[x]) == str(all_answer_key[x]) else 0 for x in range(len(self.answer_key))]
        score = round(round((10 / len(self.answer_key)), 3) * sum(grading), 2)
        cv2.putText(preview_img, f"Score: {score}/10", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.putText(preview_img, "SBD : " + sbd, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.putText(preview_img, "MDT : " + mdt, (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        self.img_result = preview_img
        self._show_image(preview_img)
        result_text = (
            f"📌 Số Báo Danh: {sbd}\n"
            f"📝 Mã Đề Thi: {mdt}\n"
            f"🎯 Điểm: {score}/10\n"
            f"✏️ Đáp Án: {', '.join(all_answer_key)}"
        )
        self.result_text.delete(1.0, 'end')
        self.result_text.insert('end', result_text)
        self.status_bar.config(text="Đã chấm xong!")

    def _show_image(self, cv_image):
        if cv_image is None:
            self.image_label.configure(image='', text='Chưa có ảnh preview', font=("Segoe UI", 14), anchor='center')
            return
        frame_width = self.image_label.winfo_width() or 800
        frame_height = self.image_label.winfo_height() or 600
        h, w = cv_image.shape[:2]
        aspect = w / h
        if aspect > frame_width / frame_height:
            new_w = frame_width
            new_h = int(frame_width / aspect)
        else:
            new_h = frame_height
            new_w = int(frame_height * aspect)
        resized = cv2.resize(cv_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.image_label.configure(image=imgtk, text='')
        self.image_label.image = imgtk

    def _save_result_image(self):
        if self.img_result is None:
            messagebox.showwarning("Cảnh báo", "Chưa có ảnh kết quả để lưu!")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
        if not file_path:
            return
        cv2.imwrite(file_path, self.img_result)
        self.status_bar.config(text=f"Đã lưu ảnh kết quả: {file_path}")
        messagebox.showinfo("Thành Công", f"Đã lưu ảnh kết quả vào {file_path}!")

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = TracNghiemGraderGUI()
    app.run()
