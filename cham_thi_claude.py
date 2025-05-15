import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from PIL import Image, ImageTk

class MCQGradingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Phần mềm chấm thi trắc nghiệm")
        self.root.geometry("1280x720")
        
        # Biến để lưu trữ đường dẫn
        self.answer_key_path = None
        self.exam_image_path = None
        self.result_image = None
        self.answer_key_image = None  # Thêm biến để lưu ảnh đáp án
        
        # Thiết lập style
        self.style = ttk.Style("darkly")
        
        # Frame chính
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=BOTH, expand=YES, padx=20, pady=20)
        
        # Frame bên trái - Các nút chức năng
        left_frame = ttk.LabelFrame(main_frame, text="Các chức năng", padding=15)
        left_frame.pack(side=LEFT, fill=BOTH, expand=NO, padx=(0, 10))
        
        # Các nút chức năng
        self.btn_select_answer = ttk.Button(
            left_frame, 
            text="Quét ảnh chấm thi", 
            command=self.select_answer_key,
            bootstyle=SUCCESS,
            width=20
        )
        self.btn_select_answer.pack(pady=10, fill=X)
        
        self.answer_key_label = ttk.Label(left_frame, text="Chưa chọn ảnh đáp án")
        self.answer_key_label.pack(pady=(0, 10), fill=X)
        
        self.btn_select_exam = ttk.Button(
            left_frame, 
            text="Chọn ảnh chấm điểm", 
            command=self.select_exam_image,
            bootstyle=INFO,
            width=20
        )
        self.btn_select_exam.pack(pady=10, fill=X)
        
        self.exam_image_label = ttk.Label(left_frame, text="Chưa chọn ảnh bài thi")
        self.exam_image_label.pack(pady=(0, 10), fill=X)
        
        self.btn_grade = ttk.Button(
            left_frame, 
            text="Chấm điểm", 
            command=self.grade_exam,
            bootstyle=PRIMARY,
            width=20
        )
        self.btn_grade.pack(pady=10, fill=X)
        
        self.btn_save = ttk.Button(
            left_frame, 
            text="Lưu ảnh chấm thi", 
            command=self.save_result,
            bootstyle=SECONDARY,
            width=20
        )
        self.btn_save.pack(pady=10, fill=X)
        
        # Thông tin kết quả
        self.result_frame = ttk.LabelFrame(left_frame, text="Kết quả", padding=10)
        self.result_frame.pack(pady=10, fill=X)
        
        self.score_label = ttk.Label(self.result_frame, text="Điểm: --/10")
        self.score_label.pack(pady=5)
        
        self.sbd_label = ttk.Label(self.result_frame, text="SBD: ---")
        self.sbd_label.pack(pady=5)
        
        self.mdt_label = ttk.Label(self.result_frame, text="MDT: ---")
        self.mdt_label.pack(pady=5)
        
        # Frame bên phải - Hiển thị hình ảnh
        self.right_frame = ttk.LabelFrame(main_frame, text="Hiển thị ảnh", padding=15)
        self.right_frame.pack(side=RIGHT, fill=BOTH, expand=YES)
        
        # Canvas để hiển thị ảnh
        self.canvas = tk.Canvas(self.right_frame, bg="lightgray")
        self.canvas.pack(fill=BOTH, expand=YES)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Sẵn sàng")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def process_answer_sheet(self, answer_image):
        """Xử lý ảnh đáp án để trích xuất các đáp án"""
        try:
            # Đây sẽ là nơi để xử lý ảnh đáp án và trích xuất đáp án
            # Trong ví dụ này, chúng ta giả định một mẫu đáp án cố định
            # Trong thực tế, bạn sẽ cần triển khai thuật toán quét đáp án từ ảnh
            
            # Ví dụ mẫu:
            # 1. Tách vùng chứa đáp án trong ảnh
            # 2. Nhận diện các ô được đánh dấu
            # 3. Chuyển đổi thành mảng đáp án
            
            # Đây là phần giả lập, thực tế sẽ cần thuật toán phức tạp hơn
            answer_key = ["A", "B", "C", "D"] * 30
            
            # Lưu ảnh đáp án đã xử lý (nếu cần)
            self.answer_key_image = answer_image
            
            return answer_key
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể xử lý ảnh đáp án: {str(e)}")
            return None
    
    def select_answer_key(self):
        """Chọn file đáp án"""
        file_path = filedialog.askopenfilename(
            title="Chọn file đáp án",
            filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            self.answer_key_path = file_path
            filename = os.path.basename(file_path)
            self.answer_key_label.config(text=f"Đã chọn: {filename}")
            self.status_var.set(f"Đã chọn file đáp án: {filename}")
            
            # Nếu là file ảnh, xử lý ảnh đáp án
            if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                img = cv2.imread(file_path)
                if img is not None:
                    self.process_answer_sheet(img)
                    self.display_image(file_path)
    
    def select_exam_image(self):
        """Chọn ảnh bài thi cần chấm"""
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh bài thi",
            filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
        )
        
        if file_path:
            self.exam_image_path = file_path
            filename = os.path.basename(file_path)
            self.exam_image_label.config(text=f"Đã chọn: {filename}")
            self.status_var.set(f"Đã chọn ảnh bài thi: {filename}")
            
            # Hiển thị ảnh lên canvas
            self.display_image(file_path)
    
    def display_image(self, image_path=None, cv_image=None):
        """Hiển thị ảnh lên canvas"""
        if image_path:
            # Đọc ảnh từ đường dẫn
            pil_image = Image.open(image_path)
        elif cv_image is not None:
            # Chuyển từ OpenCV sang PIL
            cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cv_image_rgb)
        else:
            return
        
        # Điều chỉnh kích thước ảnh để vừa với canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:  # Canvas chưa được render
            self.root.update_idletasks()
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
        
        img_width, img_height = pil_image.size
        scale = min(canvas_width/img_width, canvas_height/img_height)
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        resized_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
        
        # Chuyển sang định dạng Tkinter
        self.tk_image = ImageTk.PhotoImage(resized_image)
        
        # Xóa canvas và hiển thị ảnh mới
        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_width//2, canvas_height//2, 
            image=self.tk_image, anchor=tk.CENTER
        )
    
    def load_answer_key(self):
        """Đọc đáp án từ file"""
        if not self.answer_key_path:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn file đáp án trước!")
            return None
        
        # Nếu file đáp án là hình ảnh và đã được xử lý
        if hasattr(self, 'answer_key_image') and self.answer_key_image is not None:
            # Trả về đáp án đã xử lý từ ảnh
            return self.process_answer_sheet(self.answer_key_image)
        
        # Nếu là file text
        try:
            with open(self.answer_key_path, 'r') as file:
                content = file.read().strip()
                # Xử lý đáp án từ file, ví dụ: "ABCD,ABCD,..."
                answers = content.replace(" ", "").replace("\n", "").split(",")
                answer_key = []
                for ans in answers:
                    answer_key.extend(list(ans))
                return answer_key
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể đọc file đáp án: {str(e)}")
            return None
    
    def grade_exam(self):
        """Chấm điểm bài thi"""
        if not self.exam_image_path:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn ảnh bài thi trước!")
            return
        
        answer_key = self.load_answer_key()
        if answer_key is None:
            # Sử dụng đáp án mặc định nếu không có ảnh đáp án
            answer_key = ["A", "B", "C", "D"] * 30
            self.status_var.set("Không có ảnh đáp án, đang sử dụng đáp án mẫu...")
        else:
            self.status_var.set("Đang sử dụng đáp án từ ảnh đã chọn...")
        
        self.root.update_idletasks()
        
        try:
            # Đọc ảnh
            img = cv2.imread(self.exam_image_path)
            if img is None:
                raise Exception("Không thể đọc file ảnh bài thi")
                
            img_height, img_width, img_channels = img.shape

            # Vùng cắt theo 4 cột
            crop_regions = [
                (0, 790, 448, 2555),   # Câu 1-30
                (448, 790, 896, 2555),  # Câu 31-60
                (896, 790, 1344, 2555), # Câu 61-90
                (1344, 790, 1792, 2555) # Câu 91-120
            ]
            
            crop_sbd = (int(1281), int(154), int(1550), int(821))
            crop_img_sbd = img[crop_sbd[1]:crop_sbd[3], crop_sbd[0]:crop_sbd[2]]
            sbd, image_sbd = self.get_sbd_blue(crop_img_sbd)

            crop_mdt = (int(1558), int(154), int(1726), int(821))
            crop_img_mdt = img[crop_mdt[1]:crop_mdt[3], crop_mdt[0]:crop_mdt[2]]
            mdt, image_mdt = self.get_mdt_blue(crop_img_mdt)

            all_answer_key = []
            for idx, (x1, y1, x2, y2) in enumerate(crop_regions):
                crop_img = img[y1:y2, x1:x2]
                ans, processed_img = self.get_result_trac_nghiem(crop_img, answer_key[idx * 30: (idx + 1) * 30])
                all_answer_key.extend(ans)
                
                # Vẽ khung vùng nhận diện lên ảnh gốc
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
                img[y1:y2, x1:x2] = processed_img

            # Kiểm tra và điều chỉnh độ dài của all_answer_key
            if len(all_answer_key) < len(answer_key):
                all_answer_key.extend(["N"] * (len(answer_key) - len(all_answer_key)))

            # Tính điểm
            grading = [1 if str(answer_key[x]) == str(all_answer_key[x]) else 0 for x in range(len(answer_key))]
            score = round(round((10 / len(answer_key)), 3) * sum(grading), 2)

            # Hiện điểm trên ảnh
            string_sbd = ''.join(map(str, sbd))
            string_mdt = ''.join(map(str, mdt))
            cv2.putText(img, f"Score: {score}/10", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            cv2.putText(img, "SBD : " + string_sbd, (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            cv2.putText(img, "MDT : " + string_mdt, (50, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

            # Lưu kết quả và hiển thị
            self.result_image = img
            self.display_image(cv_image=img)
            
            # Cập nhật thông tin kết quả
            self.score_label.config(text=f"Điểm: {score}/10")
            self.sbd_label.config(text=f"SBD: {string_sbd}")
            self.mdt_label.config(text=f"MDT: {string_mdt}")
            
            self.status_var.set(f"Chấm điểm hoàn tất. Điểm: {score}/10")
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Có lỗi xảy ra khi chấm điểm: {str(e)}")
            self.status_var.set("Có lỗi xảy ra khi chấm điểm")
    
    def save_result(self):
        """Lưu kết quả chấm điểm ra file ảnh"""
        if self.result_image is None:
            messagebox.showwarning("Cảnh báo", "Chưa có kết quả chấm điểm để lưu!")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Lưu kết quả chấm điểm",
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                cv2.imwrite(file_path, self.result_image)
                messagebox.showinfo("Thông báo", f"Đã lưu kết quả thành công tại:\n{file_path}")
                self.status_var.set(f"Đã lưu kết quả tại: {file_path}")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể lưu file: {str(e)}")
                self.status_var.set("Có lỗi xảy ra khi lưu file")
    
    def get_result_trac_nghiem(self, image_trac_nghiem, ANSWER_KEY):
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
            self.status_var.set("❌ Không tìm thấy đủ ô tròn! Hãy kiểm tra ảnh đầu vào.")
            return [], image

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

    def get_sbd_blue(self, image):
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

    def get_mdt_blue(self, image):
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

# Khởi chạy ứng dụng
if __name__ == "__main__":
    root = ttk.Window()
    app = MCQGradingApp(root)
    root.mainloop()