import cv2
import os
import time

# --- KONFIGURASI ---
# Path otomatis: raw\custom\neutral
DATA_PATH = os.path.join("data", "raw", "custom") 
ACTION = "neutral"          
NO_VIDEOS = 50              
FRAMES_PER_VIDEO = 30       
# -------------------

# Buat folder
folder_path = os.path.join(DATA_PATH, ACTION)
os.makedirs(folder_path, exist_ok=True)

cap = cv2.VideoCapture(1)
# Setting resolusi (opsional)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def draw_text_center(frame, text, y_pos, bg_color=(0, 0, 0), txt_color=(255, 255, 255), scale=0.8):
    """Fungsi helper untuk bikin teks dengan background kotak agar mudah dibaca"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, scale, 2)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    
    # Gambar kotak background
    box_coords = ((text_x - 10, y_pos - 30), (text_x + text_size[0] + 10, y_pos + 10))
    cv2.rectangle(frame, box_coords[0], box_coords[1], bg_color, -1)
    
    # Tulis teks
    cv2.putText(frame, text, (text_x, y_pos), font, scale, txt_color, 2)

# ================= TAHAP 1: PREVIEW MODE (BERCERMIN) =================
print("Kamera terbuka. Silakan bercermin. Tekan SPASI di window kamera untuk mulai.")
while True:
    ret, frame = cap.read()
    if not ret: break

    # JANGAN FLIP CAMERA (Sesuai request)
    # frame = cv2.flip(frame, 1) <--- Baris ini dimatikan

    # Instruksi di layar
    draw_text_center(frame, "MODE PREVIEW (BERCERMIN)", 50, bg_color=(200, 0, 0)) # Biru
    draw_text_center(frame, "Tekan SPASI untuk Mulai Merekam", 400, bg_color=(0, 0, 0))

    cv2.imshow('Perekam Data BISINDO', frame)
    
    # Tunggu tombol SPASI (ASCII 32)
    if cv2.waitKey(1) & 0xFF == 32:
        break

# ================= TAHAP 2: HITUNG MUNDUR AWAL (3 DETIK) =================
start_time = time.time()
while (time.time() - start_time) < 3:
    ret, frame = cap.read()
    if not ret: break
    
    countdown = 3 - int(time.time() - start_time)
    
    # Tampilkan angka besar di tengah
    draw_text_center(frame, f"MULAI DALAM: {countdown}", 240, bg_color=(0, 0, 255), scale=2.0)
    cv2.imshow('Perekam Data BISINDO', frame)
    cv2.waitKey(1)

# ================= TAHAP 3: LOOP REKAM 50 VIDEO =================
for video_num in range(NO_VIDEOS):
    
    # --- FASE JEDA / GANTI GAYA (2 Detik) ---
    # Tentukan pesan berdasarkan urutan video (supaya kamu ingat ganti gaya)
    if video_num < 15:
        pesan_gaya = "GAYA: DIAM (Tatap Kamera)"
    elif video_num < 35:
        pesan_gaya = "GAYA: NOISE (Garuk/Betulin Rambut)"
    else:
        pesan_gaya = "GAYA: KOSONG (Turunkan Tangan)"

    end_wait_time = time.time() + 2 # Jeda 2 detik
    while time.time() < end_wait_time:
        ret, frame = cap.read()
        if not ret: break
        
        # Tampilkan Info Video ke-berapa
        draw_text_center(frame, f"Video {video_num+1} dari {NO_VIDEOS}", 50, bg_color=(50, 50, 50))
        # Tampilkan Instruksi Ganti Gaya
        draw_text_center(frame, "SIAP-SIAP...", 200, bg_color=(0, 165, 255), scale=1.0)
        draw_text_center(frame, pesan_gaya, 250, bg_color=(0, 100, 0), scale=0.7)
        
        cv2.imshow('Perekam Data BISINDO', frame)
        cv2.waitKey(1)
    
    # --- FASE REKAM (30 Frame) ---
    save_path = os.path.join(folder_path, f"{ACTION}_{video_num}.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(save_path, fourcc, 30.0, (1280, 720))
    
    for frame_num in range(FRAMES_PER_VIDEO):
        ret, frame = cap.read()
        if not ret: break
        
        # Simpan frame MURNI (tanpa tulisan) ke file
        out.write(frame)
        
        # Tampilkan frame DENGAN tulisan ke layar
        display_frame = frame.copy()
        cv2.circle(display_frame, (30, 30), 15, (0, 0, 255), -1) # Titik Merah (Rec)
        cv2.putText(display_frame, "REC", (55, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        draw_text_center(display_frame, f"Video {video_num+1}: {pesan_gaya}", 450, scale=0.6)
        
        cv2.imshow('Perekam Data BISINDO', display_frame)
        cv2.waitKey(1)
        
    out.release()
    print(f"Video {video_num} tersimpan.")

cap.release()
cv2.destroyAllWindows()
print("Selesai! Cek folder raw/custom/neutral.")