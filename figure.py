import cv2
import numpy as np
import os
import threading
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk
import time  # 新增：用于在采集样本时避免忙等待

class FaceRecognitionApp :
    # --- 配置常量 ---
    CONFIG_FILE = 'config.txt'
    CASCADE_FILE = 'haarcascade_frontalface_default.xml'
    TRAINING_DATA_DIR = 'data'
    TRAINER_FILE = 'trainer.yml'
    CAMERA_INDEX = 0  # 摄像头ID
    SAMPLE_COUNT = 100  # 采集样本数量
    RECOGNITION_CONFIDENCE_THRESHOLD = 70  # 识别置信度阈值，越小越严格 (0是完美匹配)
    STANDARD_FACE_SIZE = (200,200)  #  定义标准人脸尺寸

    def __init__ ( self, window ) :
        self.window = window
        self.window.title ( "智能人脸识别系统" )
        self.window.geometry ( '1000x700' )

        # --- 状态变量 ---
        self.id_dict = {}
        self.total_face_num = 0
        self.system_lock = threading.Lock ()
        self.is_recognizing = False
        self.current_frame = None


        # --- 初始化组件 ---
        if not self._initialize_components () :
            self.window.destroy ()
            return

        self._setup_gui ()
        self._load_config ()

        # --- 启动视频流 ---
        self.video_loop ()
        self.window.protocol ( "WM_DELETE_WINDOW", self.on_closing )

        # 检查是否存在训练好的模型文件，如果存在，则自动开始识别
        if os.path.exists ( self.TRAINER_FILE ) :
            self.toggle_recognition ()
        else :
            self.update_status ( "未找到训练模型，请先录入人脸。", "orange" )
        # =================================================================

    def _initialize_components ( self ) :
        """加载所有必要的OpenCV组件和文件"""
        if not os.path.exists ( self.CASCADE_FILE ) :
            messagebox.showerror ( "错误", f"关键文件丢失: {self.CASCADE_FILE}" )
            return False
        self.face_cascade = cv2.CascadeClassifier ( self.CASCADE_FILE )
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.camera = cv2.VideoCapture ( self.CAMERA_INDEX )
        if not self.camera.isOpened () :
            messagebox.showerror ( "摄像头错误",
                                f"无法打开ID为 {self.CAMERA_INDEX} 的摄像头。请检查摄像头是否连接或被其他程序占用。" )
            return False
        if not os.path.exists ( self.TRAINING_DATA_DIR ) :
            os.makedirs ( self.TRAINING_DATA_DIR )
        return True

    def _load_config ( self ) :
        """从config.txt加载人脸ID和姓名"""
        try :
            if not os.path.exists ( self.CONFIG_FILE ) :
                with open ( self.CONFIG_FILE, 'w' ) as f :
                    f.write ( '0\n' )  # 初始状态没有用户
            with open ( self.CONFIG_FILE, 'r' ) as f :
                lines = f.readlines ()
                if not lines :  # 文件可能为空
                    self.total_face_num = 0
                    print ( "配置文件为空，无用户数据。" )
                    return

                try :
                    self.total_face_num = int ( lines[0].strip () )
                except ValueError :  # 第一行不是数字
                    print ( "配置文件第一行格式错误，重置用户数量为0。" )
                    self.total_face_num = 0

                self.id_dict = {}  # 清空旧字典
                for line in lines[1 :] :  # 从第二行开始读取用户
                    line = line.strip ()
                    if line :
                        parts = line.split ( ' ', 1 )
                        if len ( parts ) == 2 and parts[0].isdigit () :
                            self.id_dict[int ( parts[0] )] = parts[1]
                        else :
                            print ( f"配置文件中发现格式错误的行: {line}" )
            print ( "配置加载完成:", self.id_dict )
        except (IOError, ValueError) as e :
            self.update_status ( f"配置文件读取错误: {e}", "red" )
            print ( f"配置文件读取错误: {e}" )

    def _setup_gui ( self ) :
        """创建图形用户界面"""
        self.status_var = tk.StringVar ()
        self.status_label = tk.Label ( self.window, textvariable=self.status_var, bg='green', fg='white',
                                    font=('Arial', 14), height=3 )
        self.status_label.pack ( fill=tk.X )
        self.update_status ( "欢迎使用人脸识别系统" )
        self.video_panel = tk.Label ( self.window )
        self.video_panel.pack ( pady=10 )
        control_frame = tk.Frame ( self.window )
        control_frame.pack ( pady=10 )
        self.btn_recognize = tk.Button ( control_frame, text='开始识别', font=('Arial', 12), width=15, height=2,
                                        command=self.toggle_recognition )
        self.btn_recognize.grid ( row=0, column=0, padx=20 )
        self.btn_enroll = tk.Button ( control_frame, text='录入新人脸', font=('Arial', 12), width=15, height=2,
                                    command=self.enroll_face )
        self.btn_enroll.grid ( row=0, column=1, padx=20 )
        self.btn_exit = tk.Button ( control_frame, text='退出系统', font=('Arial', 12), width=15, height=2,
                                    command=self.on_closing )
        self.btn_exit.grid ( row=0, column=2, padx=20 )

    def video_loop ( self ) :
        """主视频循环，用于捕获和显示帧"""
        success, frame = self.camera.read ()
        if success :
            self.current_frame = frame.copy ()  # 使用 copy() 避免在多线程中修改同一帧
            if self.is_recognizing :
                self._recognize_faces_in_frame ()

            # 缩放图像以适应窗口大小，避免图像过大导致显示问题
            h, w, _ = self.current_frame.shape
            max_width = 640
            max_height = 480
            if w > max_width or h > max_height :
                scale = min ( max_width / w, max_height / h )
                self.current_frame = cv2.resize ( self.current_frame, (int ( w * scale ), int ( h * scale )) )

            cv2image = cv2.cvtColor ( self.current_frame, cv2.COLOR_BGR2RGBA )
            img = Image.fromarray ( cv2image )
            imgtk = ImageTk.PhotoImage ( image=img )
            self.video_panel.imgtk = imgtk
            self.video_panel.config ( image=imgtk )
        self.window.after ( 15, self.video_loop )

    def toggle_recognition ( self ) :
        """切换人脸识别状态"""
        if self.is_recognizing :
            self.is_recognizing = False
            self.btn_recognize.config ( text="开始识别" )
            self.update_status ( "识别已停止" )
        else :
            if not os.path.exists ( self.TRAINER_FILE ) :
                self.update_status ( "错误: 未找到训练模型，请先录入人脸。", "red" )
                return
            try :
                self.recognizer.read ( self.TRAINER_FILE )
            except cv2.error as e :
                self.update_status ( f"错误: 无法加载训练模型文件 '{self.TRAINER_FILE}'。请尝试重新训练。详细: {e}",
                                    "red" )
                return

            self.is_recognizing = True
            self.btn_recognize.config ( text="停止识别" )
            self.update_status ( "正在进行人脸识别..." )

    def _recognize_faces_in_frame ( self ) :
        """在当前帧中识别人脸，并绘制框和文本"""
        if self.current_frame is None : return

        # 识别前先将帧转换为灰度图
        gray = cv2.cvtColor ( self.current_frame, cv2.COLOR_BGR2GRAY )

        # 提高人脸检测的鲁棒性
        faces = self.face_cascade.detectMultiScale (
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50),  # 最小人脸尺寸可以适当调大，减少误识别
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces :
            # 确保检测到的人脸区域有效
            if w == 0 or h == 0 :
                continue

            # 识别时也需要将检测到的人脸区域缩放到标准尺寸
            face_for_recognition = cv2.resize(gray[y:y+h, x:x+w], self.STANDARD_FACE_SIZE)
            id_num, confidence = self.recognizer.predict(face_for_recognition)

            # LBPH的confidence值越小，代表越可信。我们将其转换为百分比形式。
            confidence_percent = round ( 100 - confidence )

            if confidence_percent > (100 - self.RECOGNITION_CONFIDENCE_THRESHOLD) :
                name = self.id_dict.get ( id_num, "未知" )
                display_text = f"{name} ({confidence_percent}%)"
                color = (0, 255, 0)  # 绿色代表识别成功
            else :
                display_text = "陌生人"
                color = (0, 0, 255)  # 红色代表陌生人

            # 绘制矩形框
            cv2.rectangle ( self.current_frame, (x, y), (x + w, y + h), color, 2 )
            # 绘制文本
            cv2.putText ( self.current_frame, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2 )

    def enroll_face ( self ) :
        """启动录入新人脸的线程"""
        if not self.system_lock.acquire ( blocking=False ) :
            self.update_status ( "系统正忙（可能正在进行识别或录入），请稍后再试...", "orange" )
            return
        if self.is_recognizing :
            self.toggle_recognition ()
        new_name = simpledialog.askstring ( "录入新用户", "请输入您的姓名:" )
        if not new_name :
            self.update_status ( "录入已取消" )
            self.system_lock.release ()
            return

        # 获取下一个可用的ID
        next_id = 0
        if self.id_dict :
            next_id = max ( self.id_dict.keys () ) + 1
        else :
            next_id = 1  # 如果字典为空，从1开始

        enroll_thread = threading.Thread ( target=self._enroll_thread_func, args=(next_id, new_name) )
        enroll_thread.daemon = True
        enroll_thread.start ()

    def _enroll_thread_func(self, new_id, new_name):
        """在子线程中执行人脸采集和训练 (带健壮性检查)"""
        try:
            self.update_status(f"准备为 {new_name} 采集样本，请正对摄像头...")
            collected_samples = 0

            max_attempts = self.SAMPLE_COUNT * 10
            current_attempt = 0

            while collected_samples < self.SAMPLE_COUNT and current_attempt < max_attempts:
                current_attempt += 1

                if self.current_frame is None:
                    time.sleep(0.05)
                    continue

                gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    # 确保检测到的区域有有效的宽高，且尺寸不低于某个阈值（例如20x20像素）
                    if w > 20 and h > 20:
                        face_roi = gray[y:y+h, x:x+w]

                        # 在保存前，检查裁剪出的图像是否为空。
                        if face_roi.size == 0 or face_roi.shape[0] < 20 or face_roi.shape[1] < 20:
                            print(f"警告：跳过一个无效或过小的人脸切片 (尺寸: {face_roi.shape if face_roi.size > 0 else 'empty'})。")
                            continue

                        # 缩放人脸到标准尺寸
                        processed_face = cv2.resize(face_roi, self.STANDARD_FACE_SIZE)

                        collected_samples += 1

                        file_path = os.path.join(self.TRAINING_DATA_DIR, f"User.{new_id}.{collected_samples}.jpg")
                        cv2.imwrite(file_path, processed_face) # 保存处理过的人脸

                        cv2.rectangle(self.current_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                        cv2.putText(self.current_frame, f"采集中: {collected_samples}/{self.SAMPLE_COUNT}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

                        progress = int((collected_samples / self.SAMPLE_COUNT) * 100)
                        self.update_status(f"正在采集... {progress}%", schedule=True)

                        if collected_samples >= self.SAMPLE_COUNT:
                            break

            if collected_samples < self.SAMPLE_COUNT :
                if collected_samples == 0 :
                    self.update_status ( "错误: 未能采集到任何有效人脸样本，无法训练模型。请确保人脸正对摄像头。", "red",
                                        schedule=True )
                    self.system_lock.release ()
                    return
                else :
                    self.update_status (
                        f"警告: 未能采集到足够的人脸样本 ({collected_samples}/{self.SAMPLE_COUNT})。将使用现有样本进行训练。",
                        "orange", schedule=True )

            self.update_status ( "样本采集完成，开始训练模型...", schedule=True )
            # 训练模型
            self._train_model ()

            # 更新配置
            self.id_dict[new_id] = new_name
            # 根据id_dict更新total_face_num，确保其与实际用户数一致
            self.total_face_num = len ( self.id_dict )
            self._update_config_file ()

            self.update_status ( f"用户 {new_name} 录入成功！", "blue", schedule=True )

        except Exception as e :
            import traceback
            traceback.print_exc ()  # 打印完整的错误堆栈，方便调试
            self.update_status ( f"录入过程中发生错误: {e}", "red", schedule=True )
        finally :
            # 确保锁一定会被释放，防止死锁
            if self.system_lock.locked () :
                self.system_lock.release ()

    def _train_model ( self ) :
        """使用data目录下的所有图片训练一个统一的模型"""
        image_paths = [os.path.join ( self.TRAINING_DATA_DIR, f ) for f in os.listdir ( self.TRAINING_DATA_DIR )]

        # 过滤掉非图像文件或格式不匹配的文件
        valid_image_paths = [p for p in image_paths if
                            os.path.isfile ( p ) and p.lower ().endswith ( ('.png', '.jpg', '.jpeg') )]

        face_samples = []
        ids = []

        problematic_images_count = 0

        for image_path in valid_image_paths :
            try :
                img = Image.open ( image_path ).convert ( 'L' )  # 确保转换为灰度图 'L'
                img_np = np.array ( img, 'uint8' )
                #img_np = cv2.imread ( image_path, cv2.IMREAD_GRAYSCALE )

                # 检查图像是否为空或尺寸过小
                if img_np.size == 0 or img_np.shape[0] < 20 or img_np.shape[1] < 20 :
                    print (
                        f"警告: 跳过空或尺寸过小的图像文件 (尺寸: "
                        f"{img_np.shape if img_np.size > 0 else 'empty'}): {image_path}" )
                    problematic_images_count += 1
                    continue

                # 即使文件系统中的图片尺寸不一致，这里也能强制统一
                img_np_resized = cv2.resize(img_np, self.STANDARD_FACE_SIZE)

                # 从文件名提取ID: User.ID.SampleNum.jpg
                filename = os.path.basename ( image_path )
                parts = filename.split ( '.' )
                if len ( parts ) >= 2 and parts[0] == 'User' and parts[1].isdigit () :
                    id_num = int ( parts[1] )
                    face_samples.append ( img_np_resized )
                    ids.append ( id_num )
                else :
                    print ( f"警告: 无法从文件名提取ID或文件名格式不正确: {image_path}" )
                    problematic_images_count += 1
            except Exception as e :
                print ( f"错误: 处理图像文件 {image_path} 时发生异常: {e}" )
                problematic_images_count += 1
                continue

        if problematic_images_count > 0 :
            self.update_status ( f"在训练前跳过了 {problematic_images_count} 个有问题的人脸样本。", "orange",
                                schedule=True )

        if not face_samples or not ids :
            print ( "没有找到足够的可训练人脸数据，或者所有数据都有问题。" )
            self.update_status ( "没有足够的数据进行人脸模型训练，请先录入人脸。", "red", schedule=True )
            return

        # 检查样本数量和ID数量是否匹配
        if len ( face_samples ) != len ( ids ) :
            print ( f"错误: 样本数量 ({len ( face_samples )}) 和ID数量 ({len ( ids )}) 不匹配。训练失败。" )
            self.update_status ( "样本数据不一致，模型训练失败。", "red", schedule=True )
            return

        # 检查是否有足够的独特标签（ID）来训练
        unique_ids = np.unique ( ids )
        if len ( unique_ids ) < 1 :
            print ( "错误: 训练数据中没有有效的用户ID。" )
            self.update_status ( "没有有效的用户ID，模型训练失败。", "red", schedule=True )
            return
        # LBPHFaceRecognizer需要至少两个样本，或者至少两个不同的ID才能进行有效训练。
        # 如果只有一个唯一的ID，且样本数量少于2，某些OpenCV版本可能会报错或训练效果很差。
        if len ( unique_ids ) == 1 and len ( face_samples ) < 2 :
            print (
                f"警告: 只有一个用户ID ({unique_ids[0]}) 且样本数量过少 ({len ( face_samples )})。建议增加样本以提高识别准确度。" )
            # 此时仍尝试训练，因为可能只是警告而非硬性错误

        print ( f"准备训练模型，共有 {len ( face_samples )} 个人脸样本，对应 {len ( unique_ids )} 个用户ID。" )

        try :
            self.recognizer.train ( face_samples, np.array ( ids ) )
            self.recognizer.save ( self.TRAINER_FILE )
            print ( "模型训练并保存成功！" )
        except cv2.error as e :
            print ( f"OpenCV 训练模型时发生错误: {e}" )
            self.update_status ( f"模型训练失败: {e} (请检查数据完整性或尝试重新录入)", "red", schedule=True )
        except Exception as e :
            print ( f"训练模型时发生未知错误: {e}" )
            self.update_status ( f"模型训练失败: {e}", "red", schedule=True )

    def _update_config_file ( self ) :
        """将内存中的id_dict和total_face_num写回文件"""
        with open ( self.CONFIG_FILE, 'w' ) as f :
            f.write ( f"{self.total_face_num}\n" )
            # 按照ID排序写入，确保一致性
            for id_num in sorted ( self.id_dict.keys () ) :
                f.write ( f"{id_num} {self.id_dict[id_num]}\n" )
        print ( "配置文件更新成功！" )

    def update_status ( self, text, color='green', schedule=False ) :
        """安全地更新状态栏文本和颜色"""

        def task () :
            self.status_var.set ( text )
            self.status_label.config ( bg=color )

        if schedule :
            self.window.after ( 0, task )
        else :
            task ()

    def on_closing ( self ) :
        """窗口关闭时的清理操作"""
        print ( "正在关闭应用..." )
        self.is_recognizing = False  # 停止识别循环
        self.camera.release ()
        self.window.destroy ()


if __name__ == "__main__" :
    root = tk.Tk ()
    app = FaceRecognitionApp ( root )
    root.mainloop ()