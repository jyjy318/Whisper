import tkinter as tk
from tkinter.filedialog import askdirectory, askopenfilename, askopenfilenames
from tkinter import messagebox, ttk
import jax.numpy as jnp
from whisper_jax import FlaxWhisperPipline
import os
import subprocess
import time
import torch
import jax
import threading

root = tk.Tk()
root.title('Whisper Jax GUI')
root.minsize(800, 500)  # 최소 사이즈
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 라이브러리 충돌 방지
os.environ['CURL_CA_BUNDLE'] = ''  # SSL 보안 회피

def run_in_background(function, args):
    thread = threading.Thread(target=function, args=args)
    thread.daemon = True
    thread.start()

def select_file():
    try:
        filename = askopenfilename(initialdir="./", filetypes=(("Video files", ".mp4"), ('All files', '*.*')))
        if filename:
            listbox1.delete('end', "end")
            listbox1.insert('end', filename)
            listbox3.insert('end', "선택된 파일 : " + filename)
    except:
        messagebox.showerror("Error", "오류가 발생했습니다.")

def whisper_jax():
    video_path = listbox1.get(0)
    if video_path == "":
        messagebox.showerror("Error", "파일을 먼저 선택해주세요.")
    else:
        audio_path = video_path.replace(".mp4", "_audio.wav")
        listbox3.insert('end',
                        "추론에 필요한 저용량 오디오파일 출력 시작 : ffmpeg -i " + video_path + " -ar 16000 -ac 1 -b:a 96K -acodec pcm_s16le " + audio_path)
        subprocess.run(
            ['D:\\ffmpeg\\ffmpeg.exe', '-i', video_path, '-ar', '16000', '-ac', '1', '-b:a', '96K', '-acodec',
             'pcm_s16le', audio_path])
        listbox2.insert(0, "변환된 오디오파일 : " + audio_path)
        run_in_background(generate_subtitle, (video_path, audio_path))

def measure_time(func):
    # 함수 실행 시간 측정 데코레이터
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        listbox3.insert('end', f"소요 시간: {end_time - start_time:.2f} 초")

        return result

    return wrapper

def format_timestamp(seconds: float, always_include_hours: bool = False, decimal_marker: str = '.'):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"

def save_srt(result, srt):
    with open(srt, 'w', encoding='utf-8') as out:
        for i, chunk in enumerate(result['chunks']):
            start = chunk['timestamp'][0]
            end = chunk['timestamp'][1]
            text = chunk['text']

            # start와 end가 None이 아닌지 확인
            if start is not None and end is not None:
                out.write(
                    f"{i + 1}\n"
                    f"{format_timestamp(start, always_include_hours=True, decimal_marker=',')} --> "
                    f"{format_timestamp(end, always_include_hours=True, decimal_marker=',')}\n"
                    f"{text.strip().replace('-->', '->')}\n\n"
                )

def delete_audio(audio_path):
    # 오디오 파일 삭제
    os.remove(audio_path)

@measure_time
def generate_subtitle(video_path, audio_path):
    torch_device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")
    jax_devices = jax.devices()
    listbox3.insert('end', f"현재 사용 중인 PyTorch 디바이스: {torch_device}")
    # PyTorch와 JAX의 디바이스가 동일한지 확인
    if str(torch_device) == str(jax_devices[0]):
        listbox3.insert('end', "PyTorch와 JAX가 동일한 디바이스를 사용 중입니다.")
    else:
        listbox3.insert('end', f"PyTorch 디바이스: {torch_device}")
        listbox3.insert('end', f"JAX 디바이스: {jax_devices[0]}")
        listbox3.insert('end', "PyTorch와 JAX가 다른 디바이스를 사용 중입니다.")
    # 자막 생성
    pipeline = FlaxWhisperPipline("openai/whisper-large-v2", dtype=jnp.float16)

    listbox3.insert('end', video_path + " 파일 large-v2모델 생성 시작 ")
    video_result = pipeline(audio_path, return_timestamps=True)  # medium jnp.float16
    listbox3.insert('end', video_path + " 파일 large-v2모델 생성 완료 ")
    print("srt 파일 저장 시작")
    listbox3.insert('end', video_path + " srt 파일 저장 시작")
    save_srt(video_result, video_path.replace(".mp4", "_largev2.srt"))
    listbox3.insert('end', video_path + " srt 파일 저장 완료")
    del pipeline
    # 오디오 파일 삭제
    delete_audio(audio_path)
    listbox3.insert('end', "오디오 파일 (" + audio_path + ") 삭제완료")

'''1. 프레임 생성'''
# 상단 프레임 (LabelFrame)
frm1 = tk.LabelFrame(root, text="변환", pady=15, padx=15)  # pad 내부
frm1.grid(row=0, column=0, pady=10, padx=10, sticky="nswe")  # pad 내부
root.columnconfigure(0, weight=1)  # 프레임 (0,0)은 크기에 맞춰 늘어나도록
root.rowconfigure(0, weight=1)
# 하단 프레임 (Frame)
frm2 = tk.Frame(root, pady=10)
frm2.grid(row=1, column=0, pady=10)
'''2. 요소 생성'''
# 레이블
lbl1 = tk.Label(frm1, text='파일 선택')
lbl2 = tk.Label(frm1, text='변환된 오디오파일')
# 리스트박스
listbox1 = tk.Listbox(frm1, width=40, height=1)
listbox2 = tk.Listbox(frm1, width=40, height=1)
listbox3 = tk.Listbox(frm1, width=60, height=10)

# 버튼
btn1 = tk.Button(frm1, text="찾아보기", width=8, command=select_file)
btn2 = tk.Button(frm1, text="자막생성", width=8, command=whisper_jax)

'''3. 요소 배치'''
# 상단 프레임
lbl1.grid(row=0, column=0, sticky="e")
lbl2.grid(row=1, column=0, sticky="e")
listbox1.grid(row=0, column=1, columnspan=2, sticky="we")
listbox2.grid(row=1, column=1, columnspan=2, sticky="we")
listbox3.grid(row=2, column=1, columnspan=2, sticky="nsew")
btn1.grid(row=0, column=3)
btn2.grid(row=1, column=3)
# 상단프레임 grid (2,1)은 창 크기에 맞춰 늘어나도록
frm1.rowconfigure(2, weight=1)
frm1.columnconfigure(1, weight=1)

# 수직 스크롤바 생성
scrollbar = ttk.Scrollbar(frm1, orient=tk.VERTICAL, command=listbox3.yview)
listbox3['yscrollcommand'] = scrollbar.set

# 가로 스크롤바 생성
scrollbar_horiz = ttk.Scrollbar(frm1, orient=tk.HORIZONTAL, command=listbox3.xview)
listbox3['xscrollcommand'] = scrollbar_horiz.set

# 스크롤바 배치
scrollbar.grid(row=2, column=2, rowspan=2, sticky="ns")
scrollbar_horiz.grid(row=3, column=1, columnspan=2, sticky="ew")

# 스크롤바 - 기능 연결
scrollbar = tk.Scrollbar(frm1)
scrollbar.config(command=listbox3.yview)
listbox3.config(yscrollcommand=scrollbar.set)

# 하단 프레임
'''실행'''
root.mainloop()