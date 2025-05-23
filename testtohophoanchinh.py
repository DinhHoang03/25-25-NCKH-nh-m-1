import cv2
import numpy as np

def get_result_trac_nghiem(image_trac_nghiem, ANSWER_KEY):
    translate = {"A": 0, "B": 1, "C": 2, "D": 3}
    revert_translate = {0: "A", 1: "B", 2: "C", 3: "D", -1: "N"}
    
    image = image_trac_nghiem.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=50, param2=45, minRadius=10, maxRadius=20)
    
    questionCnts = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            questionCnts.append((x, y, r))
    
    if len(questionCnts) < 120:
        print("⚠ Không tìm thấy đủ 120 ô tròn. Thử lại với phương pháp khác...")
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

if __name__ == "__main__":
    # Đọc ảnh
    link = "anhco1test.jpg"
    img = cv2.imread(link)
    img_height, img_width, img_channels = img.shape

    ANSWER_KEY = ["A", "B", "C", "D"] * 30  # Đáp án mẫu

    #Vùng cắt theo 4 cột
    crop_regions = [
        (0, 790, 448, 2555),   # Câu 1-30
        (448, 790, 896, 2555),  # Câu 31-60
        (896, 790, 1344, 2555), # Câu 61-90
        (1344, 790, 1792, 2555) # Câu 91-120
    ]
    # crop_regions = [
    #     (0, 770, 430, 2555),   # Câu 1-30
    #     (430, 770, 860, 2555),  # Câu 31-60
    #     (860, 770, 1290, 2555), # Câu 61-90
    #     (1290, 770, 1720, 2555) # Câu 91-120
    # ]
    crop_sbd = (int(1281),
                int(154),
                int(1550),
                int(821))
    crop_img_sbd = img[crop_sbd[1]:crop_sbd[3], crop_sbd[0]:crop_sbd[2]]
    sbd, image_sbd = get_sbd_blue(crop_img_sbd)

    crop_mdt = (int(1558),
                int(154),
                int(1726),
                int(821))
    crop_img_mdt = img[crop_mdt[1]:crop_mdt[3], crop_mdt[0]:crop_mdt[2]]

    mdt, image_mdt = get_mdt_blue(crop_img_mdt)

    all_answer_key = []
    for idx, (x1, y1, x2, y2) in enumerate(crop_regions):
        crop_img = img[y1:y2, x1:x2]
        ans, processed_img = get_result_trac_nghiem(crop_img, ANSWER_KEY[idx * 30: (idx + 1) * 30])
        all_answer_key.extend(ans)
        
        # Vẽ khung vùng nhận diện lên ảnh gốc
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        img[y1:y2, x1:x2] = processed_img

    # Kiểm tra và điều chỉnh độ dài của all_answer_key
    if len(all_answer_key) < len(ANSWER_KEY):
        all_answer_key.extend(["N"] * (len(ANSWER_KEY) - len(all_answer_key)))

    # Tính điểm
    grading = [1 if str(ANSWER_KEY[x]) == str(all_answer_key[x]) else 0 for x in range(len(ANSWER_KEY))]
    score = round(round((10 / len(ANSWER_KEY)), 3) * sum(grading), 2)

    # Hiện điểm trên ảnh
    string_sbd = ''.join(map(str, sbd))
    string_mdt = ''.join(map(str, mdt))
    cv2.putText(img, f"Score: {score}/10", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    cv2.putText(img, "SBD : " + string_sbd, (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    cv2.putText(img, "MDT : " + string_mdt, (50, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    imS = cv2.resize(img, (800, 800))
    cv2.imshow("Result", imS)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
