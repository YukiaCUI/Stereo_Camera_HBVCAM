import cv2

# 打开拼接的双目摄像头
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("无法获取摄像头图像")
        break

    # 假设拼接图像的分辨率是 2560x720
    # 根据实际分辨率调整
    height, width, _ = frame.shape
    left_frame = frame[:, :width // 2]
    right_frame = frame[:, width // 2:]

    # 显示左右两个图像
    cv2.imshow('Left Camera', left_frame)
    cv2.imshow('Right Camera', right_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源
cap.release()
cv2.destroyAllWindows()
