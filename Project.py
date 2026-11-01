import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageOps, ImageEnhance
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# --- CONFIGURATION ---
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

class ModernFilterApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window Setup
        self.title("PhotoFilter Pro")
        self.geometry("1280x850")
        self.minsize(1000, 700)
        
        # Intercept Close Button
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # --- State ---
        self.original_image = None 
        self.processed_image = None 
        self.sidebar_visible = True 
        
        # Crop State
        self.crop_mode = False
        self.crop_start_x = 0
        self.crop_start_y = 0
        
        # --- Mathematical Kernels ---
        self.kernels = {
            "Identity (No Effect)": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
            "Box Blur": np.ones((3, 3)) / 9.0,
            "Sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
            "Edge Detection": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
            "Emboss": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]),
            "Sobel Left (Vertical)": np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]),
            "Sobel Top (Horizontal)": np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        }

        # --- PAGE CONTAINER ---
        # Using Tuples for colors: (Light, Dark)
        self.container = ctk.CTkFrame(self, fg_color=("#f3f4f6", "#111827")) 
        self.container.pack(fill="both", expand=True)

        self._init_welcome_screen()
        self._init_editor_screen()
        self._init_toast_notification()

        # Start
        self.show_welcome_screen()
        self.bind('<Configure>', self._on_resize)

    # ==========================================
    # 1. MODERN WELCOME SCREEN
    # ==========================================
    def _init_welcome_screen(self):
        self.welcome_frame = ctk.CTkFrame(self.container, fg_color=("#f3f4f6", "#111827"))
        
        center_box = ctk.CTkFrame(self.welcome_frame, fg_color="transparent")
        center_box.place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(center_box, text="PhotoFilter", font=("Roboto Medium", 40), text_color=("#1e293b", "#f8fafc")).pack(pady=(0, 5))
        ctk.CTkLabel(center_box, text="Advanced Matrix Processing", font=("Roboto", 18), text_color=("#64748b", "#94a3b8")).pack(pady=(0, 40))

        self.btn_big_open = ctk.CTkButton(center_box, text="Open Image", font=("Roboto", 16, "bold"), 
                                          height=50, width=220, corner_radius=25, fg_color="#2563eb", hover_color="#1d4ed8",
                                          command=self.load_image_from_welcome)
        self.btn_big_open.pack()

    def show_welcome_screen(self):
        self.editor_frame.pack_forget()
        self.welcome_frame.pack(fill="both", expand=True)

    def load_image_from_welcome(self):
        if self.load_image(): 
            self.welcome_frame.pack_forget()
            self.editor_frame.pack(fill="both", expand=True)
            self.update_idletasks()
            self.update_display()

    # ==========================================
    # 2. EDITOR SCREEN (Main UI)
    # ==========================================
    def _init_editor_screen(self):
        self.editor_frame = ctk.CTkFrame(self.container, fg_color=("#f3f4f6", "#111827"))
        
        # Grid Configuration
        self.editor_frame.grid_columnconfigure(0, weight=0) 
        self.editor_frame.grid_columnconfigure(1, weight=1) 
        self.editor_frame.grid_rowconfigure(0, weight=1)

        self._init_sidebar(self.editor_frame)
        self._init_main_area(self.editor_frame)

    def _init_sidebar(self, parent):
        self.sidebar_container = ctk.CTkFrame(parent, fg_color="transparent", corner_radius=0)
        self.sidebar_container.grid(row=0, column=0, sticky="nsew")

        self.sidebar = ctk.CTkScrollableFrame(self.sidebar_container, width=320, corner_radius=0, fg_color=("white", "#1f2937"))
        self.sidebar.pack(fill="both", expand=True)

        # Home Button & Theme Toggle
        header_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        header_frame.pack(pady=(20, 10), fill="x", padx=20)
        
        ctk.CTkButton(header_frame, text="← Home", width=60, height=24, fg_color=("#e2e8f0", "#374151"), 
                      text_color=("#334155", "#f1f5f9"), hover_color=("#cbd5e1", "#4b5563"), corner_radius=6,
                      command=self.show_welcome_screen).pack(side="left")

        # --- DARK MODE SWITCH ---
        self.theme_switch = ctk.CTkSwitch(header_frame, text="Dark", command=self.change_appearance_mode, font=("Roboto", 12))
        self.theme_switch.pack(side="right")

        # Title
        ctk.CTkLabel(self.sidebar, text="Tools", font=("Roboto Medium", 20), text_color=("#0f172a", "#f8fafc"), anchor="w").pack(padx=20, pady=(10, 5), fill="x")

        # File Actions
        self._create_section_label("File Operations")
        self.btn_upload = ctk.CTkButton(self.sidebar, text="Upload Image", height=36, fg_color="#2563eb", hover_color="#1d4ed8", command=self.load_image)
        self.btn_upload.pack(padx=20, pady=5, fill="x")
        
        action_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        action_frame.pack(padx=20, pady=5, fill="x")
        ctk.CTkButton(action_frame, text="Export", width=120, height=32, fg_color=("white", "#374151"), border_width=1, border_color="#cbd5e1", text_color=("#0f172a", "#f8fafc"), hover_color=("#f1f5f9", "#4b5563"), command=self.save_image).pack(side="left", padx=(0,5), expand=True, fill="x")
        ctk.CTkButton(action_frame, text="Reset", width=120, height=32, fg_color=("white", "#374151"), border_width=1, border_color="#fca5a5", text_color="#ef4444", hover_color=("#fef2f2", "#451a1a"), command=self.reset_image).pack(side="right", padx=(5,0), expand=True, fill="x")

        # Geometry
        self._create_section_label("Geometry")
        geo_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        geo_frame.pack(padx=20, pady=5, fill="x")
        ctk.CTkButton(geo_frame, text="⟲ -90°", width=80, fg_color=("#f1f5f9", "#374151"), text_color=("#334155", "#f1f5f9"), hover_color=("#e2e8f0", "#4b5563"), command=lambda: self.rotate_image(90)).pack(side="left", padx=2, expand=True, fill="x")
        ctk.CTkButton(geo_frame, text="+90° ⟳", width=80, fg_color=("#f1f5f9", "#374151"), text_color=("#334155", "#f1f5f9"), hover_color=("#e2e8f0", "#4b5563"), command=lambda: self.rotate_image(-90)).pack(side="left", padx=2, expand=True, fill="x")
        ctk.CTkButton(geo_frame, text="Flip", width=80, fg_color=("#f1f5f9", "#374151"), text_color=("#334155", "#f1f5f9"), hover_color=("#e2e8f0", "#4b5563"), command=self.flip_image).pack(side="left", padx=2, expand=True, fill="x")

        self.btn_crop = ctk.CTkButton(self.sidebar, text="Enable Crop Tool", fg_color="#f59e0b", hover_color="#d97706", text_color="white", command=self.toggle_crop)
        self.btn_crop.pack(padx=20, pady=5, fill="x")

        # Matrix Filters
        self._create_section_label("Matrix Filter")
        self.filter_var = ctk.StringVar(value="Identity (No Effect)")
        self.filter_dropdown = ctk.CTkComboBox(self.sidebar, values=list(self.kernels.keys()), command=self.load_kernel_values, variable=self.filter_var, height=32, state="readonly")
        self.filter_dropdown.pack(padx=20, pady=5, fill="x")

        self.matrix_frame = ctk.CTkFrame(self.sidebar, fg_color=("#f8fafc", "#1f2937"), border_width=1, border_color=("#e2e8f0", "#374151"))
        self.matrix_frame.pack(padx=20, pady=10)
        self.matrix_entries = []
        for r in range(3):
            row_entries = []
            for c in range(3):
                entry = ctk.CTkEntry(self.matrix_frame, width=50, justify="center", fg_color=("white", "#374151"), border_color=("#cbd5e1", "#4b5563"))
                entry.grid(row=r, column=c, padx=4, pady=4)
                entry.insert(0, "0")
                row_entries.append(entry)
            self.matrix_entries.append(row_entries)
        
        self.btn_apply = ctk.CTkButton(self.sidebar, text="Apply Matrix Kernel", fg_color=("#0f172a", "#2563eb"), hover_color=("#334155", "#1d4ed8"), command=self.apply_custom_matrix)
        self.btn_apply.pack(padx=20, pady=5, fill="x")

        # Post Processing
        self._create_section_label("Post-Processing")
        ctk.CTkLabel(self.sidebar, text="Brightness", font=("Arial", 12), text_color="gray").pack(anchor="w", padx=25)
        self.slider_bright = ctk.CTkSlider(self.sidebar, from_=0.5, to=2.0, command=self.adjust_color, progress_color="#2563eb")
        self.slider_bright.set(1.0)
        self.slider_bright.pack(padx=20, pady=(0, 10), fill="x")
        
        fx_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        fx_frame.pack(padx=20, pady=5, fill="x")
        ctk.CTkButton(fx_frame, text="Sepia", width=80, fg_color=("#f1f5f9", "#374151"), text_color=("#334155", "#f1f5f9"), hover_color=("#e2e8f0", "#4b5563"), command=self.apply_sepia).pack(side="left", padx=2, expand=True, fill="x")
        ctk.CTkButton(fx_frame, text="Solarize", width=80, fg_color=("#f1f5f9", "#374151"), text_color=("#334155", "#f1f5f9"), hover_color=("#e2e8f0", "#4b5563"), command=self.apply_solarize).pack(side="left", padx=2, expand=True, fill="x")
        ctk.CTkButton(fx_frame, text="Invert", width=80, fg_color=("#f1f5f9", "#374151"), text_color=("#334155", "#f1f5f9"), hover_color=("#e2e8f0", "#4b5563"), command=self.invert_colors).pack(side="left", padx=2, expand=True, fill="x")

        self.load_kernel_values("Identity (No Effect)")

    def _create_section_label(self, text):
        ctk.CTkLabel(self.sidebar, text=text.upper(), font=("Roboto", 11, "bold"), text_color="#94a3b8", anchor="w").pack(padx=20, pady=(20, 5), fill="x")

    def _init_main_area(self, parent):
        self.main_frame = ctk.CTkFrame(parent, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)

        # --- Top Bar for Main Area ---
        top_bar = ctk.CTkFrame(self.main_frame, fg_color="transparent", height=40)
        top_bar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        self.btn_menu = ctk.CTkButton(top_bar, text="☰ MENU", width=80, fg_color="transparent", 
                                      text_color=("#1e293b", "#f8fafc"), hover_color=("#e2e8f0", "#374151"), 
                                      font=("Roboto", 14, "bold"), anchor="w",
                                      command=self.toggle_sidebar)
        self.btn_menu.pack(side="left")

        ctk.CTkLabel(top_bar, text="Original", font=("Roboto Medium", 14), text_color=("#475569", "#94a3b8")).pack(side="left", expand=True, padx=(0, 100))
        ctk.CTkLabel(top_bar, text="Processed", font=("Roboto Medium", 14), text_color=("#475569", "#94a3b8")).pack(side="left", expand=True, padx=(0, 100))

        # --- Canvas Areas ---
        self.frame_orig = ctk.CTkFrame(self.main_frame, fg_color=("white", "#1f2937"), corner_radius=10, border_width=1, border_color=("#e2e8f0", "#374151"))
        self.frame_orig.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        
        # We handle Canvas background manually in update_display to match theme
        self.canvas_orig = tk.Canvas(self.frame_orig, highlightthickness=0)
        self.canvas_orig.pack(expand=True)
        self.canvas_orig.bind("<ButtonPress-1>", self.on_crop_start)
        self.canvas_orig.bind("<B1-Motion>", self.on_crop_drag)
        self.canvas_orig.bind("<ButtonRelease-1>", self.on_crop_end)

        self.frame_proc = ctk.CTkFrame(self.main_frame, fg_color=("white", "#1f2937"), corner_radius=10, border_width=1, border_color=("#e2e8f0", "#374151"))
        self.frame_proc.grid(row=1, column=1, sticky="nsew", padx=(10, 0))
        
        self.canvas_proc = tk.Canvas(self.frame_proc, highlightthickness=0)
        self.canvas_proc.pack(expand=True)

    def _init_toast_notification(self):
        self.toast = ctk.CTkLabel(self, text="", fg_color="#10b981", text_color="white", corner_radius=20, width=200, height=40, font=("Roboto", 13, "bold"))
        self.toast.place(relx=0.5, rely=-0.1, anchor="center")
    
    # ==========================================
    # LOGIC
    # ==========================================
    
    def change_appearance_mode(self):
        if self.theme_switch.get() == 1:
            ctk.set_appearance_mode("Dark")
        else:
            ctk.set_appearance_mode("Light")
        # Small delay to ensure colors update before canvas redraw
        self.after(100, self.update_display)

    def toggle_sidebar(self):
        if self.sidebar_visible:
            self.sidebar_container.grid_remove()
            self.sidebar_visible = False
        else:
            self.sidebar_container.grid()
            self.sidebar_visible = True
        self.update_idletasks()
        self.update_display()

    def update_display(self):
        if self.original_image is None: return
        
        # Match canvas background to the frame background manually
        canvas_bg = "#1f2937" if ctk.get_appearance_mode() == "Dark" else "white"
        self.canvas_orig.configure(bg=canvas_bg)
        self.canvas_proc.configure(bg=canvas_bg)

        total_w = self.main_frame.winfo_width()
        w_avail = (total_w) // 2 - 40 
        h_avail = self.main_frame.winfo_height() - 80
        
        if w_avail < 100: w_avail = 100 
        if h_avail < 100: h_avail = 100

        def get_resized_tk_image(img):
            w_img, h_img = img.size
            scale = min(w_avail/w_img, h_avail/h_img)
            new_size = (int(w_img * scale), int(h_img * scale))
            resized_pil = img.resize(new_size, Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(resized_pil), scale, new_size

        self.tk_img_orig, self.scale_factor, self.disp_size = get_resized_tk_image(self.original_image)
        w_disp, h_disp = self.disp_size
        
        self.canvas_orig.configure(width=w_disp, height=h_disp)
        self.canvas_orig.delete("all")
        self.canvas_orig.create_image(0, 0, image=self.tk_img_orig, anchor="nw")

        self.tk_img_proc, _, _ = get_resized_tk_image(self.processed_image)
        self.canvas_proc.configure(width=w_disp, height=h_disp)
        self.canvas_proc.delete("all")
        self.canvas_proc.create_image(0, 0, image=self.tk_img_proc, anchor="nw")

    # [Remaining methods: load_image, load_kernel_values, apply_custom_matrix, _convolve, toggle_crop, etc. stay exactly the same as your original code]

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.original_image = Image.open(path).convert("RGB")
            self.processed_image = self.original_image.copy()
            self.update_display()
            self.show_toast("Image Loaded")
            return True
        return False

    def load_kernel_values(self, selection):
        matrix = self.kernels.get(selection, self.kernels["Identity (No Effect)"])
        for r in range(3):
            for c in range(3):
                self.matrix_entries[r][c].delete(0, "end")
                val = matrix[r][c]
                txt = f"{int(val)}" if val == int(val) else f"{val:.2f}"
                self.matrix_entries[r][c].insert(0, txt)

    def apply_custom_matrix(self):
        if self.original_image is None: 
            self.show_toast("No Image Loaded", is_error=True)
            return
        
        try:
            custom_kernel = []
            for r in range(3):
                row = []
                for c in range(3):
                    row.append(float(self.matrix_entries[r][c].get()))
                custom_kernel.append(row)
            kernel = np.array(custom_kernel)
        except ValueError:
            self.show_toast("Invalid Matrix", is_error=True)
            return

        self.update() 
        img_array = np.array(self.original_image, dtype=np.float32)
        output = self._convolve(img_array, kernel)
        self.processed_image = Image.fromarray(np.clip(output, 0, 255).astype('uint8'))
        self.update_display()
        self.show_toast("Matrix Applied")

    def _convolve(self, image_array, kernel):
        pad = 1
        output = np.zeros_like(image_array)
        kernel_sum = np.sum(kernel)
        is_edge_detection = abs(kernel_sum) < 0.01
        
        for ch in range(3):
            channel = image_array[:, :, ch]
            padded = np.pad(channel, pad, mode='edge')
            windows = sliding_window_view(padded, (3, 3))
            convolved = np.sum(windows * kernel, axis=(2, 3))
            if is_edge_detection: convolved = np.abs(convolved)
            output[:, :, ch] = convolved
        return output

    def toggle_crop(self):
        if not self.original_image: return
        self.crop_mode = not self.crop_mode
        if self.crop_mode:
            self.btn_crop.configure(fg_color="#ef4444", text="Cancel Crop (Drag Now)", hover_color="#dc2626")
            self.show_toast("Crop Mode: Drag on Left Image")
            self.canvas_orig.configure(cursor="cross")
        else:
            self.btn_crop.configure(fg_color="#f59e0b", text="Enable Crop Tool", hover_color="#d97706")
            self.canvas_orig.configure(cursor="")
            self.canvas_orig.delete("rect")

    def on_crop_start(self, event):
        if not self.crop_mode: return
        self.crop_start_x = event.x
        self.crop_start_y = event.y
        self.canvas_orig.delete("rect")

    def on_crop_drag(self, event):
        if not self.crop_mode: return
        self.canvas_orig.delete("rect")
        self.rect_id = self.canvas_orig.create_rectangle(
            self.crop_start_x, self.crop_start_y, event.x, event.y,
            outline="#ef4444", width=2, tags="rect"
        )

    def on_crop_end(self, event):
        if not self.crop_mode: return
        end_x, end_y = event.x, event.y
        x1, x2 = sorted([self.crop_start_x, end_x])
        y1, y2 = sorted([self.crop_start_y, end_y])
        if self.scale_factor == 0: return 
        real_x1, real_y1 = int(x1 / self.scale_factor), int(y1 / self.scale_factor)
        real_x2, real_y2 = int(x2 / self.scale_factor), int(y2 / self.scale_factor)
        w, h = self.original_image.size
        real_x1, real_y1 = max(0, min(real_x1, w)), max(0, min(real_y1, h))
        real_x2, real_y2 = max(0, min(real_x2, w)), max(0, min(real_y2, h))
        if (real_x2 - real_x1) < 10 or (real_y2 - real_y1) < 10:
            self.show_toast("Selection too small", is_error=True)
            return
        self.original_image = self.original_image.crop((real_x1, real_y1, real_x2, real_y2))
        self.processed_image = self.original_image.copy()
        self.toggle_crop()
        self.update_display()
        self.show_toast("Cropped Successfully")

    def apply_sepia(self):
        if not self.original_image: return
        img = np.array(self.original_image)
        sepia_matrix = np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]])
        sepia_img = img.dot(sepia_matrix.T)
        sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
        self.processed_image = Image.fromarray(sepia_img)
        self.update_display()
        self.show_toast("Sepia Filter Applied")

    def apply_solarize(self):
        if self.original_image:
            self.processed_image = ImageOps.solarize(self.original_image, threshold=128)
            self.update_display()
            self.show_toast("Solarize Applied")

    def invert_colors(self):
        if self.processed_image:
            self.processed_image = ImageOps.invert(self.processed_image)
            self.update_display()
            self.show_toast("Colors Inverted")

    def adjust_color(self, value):
        if self.original_image:
            enhancer = ImageEnhance.Brightness(self.original_image)
            self.processed_image = enhancer.enhance(float(value))
            self.update_display()

    def reset_image(self):
        if self.original_image:
            self.processed_image = self.original_image.copy()
            self.slider_bright.set(1.0)
            self.update_display()
            self.load_kernel_values("Identity (No Effect)")
            self.show_toast("Reset to Original")

    def save_image(self):
        if self.processed_image:
            path = filedialog.asksaveasfilename(defaultextension=".png")
            if path: 
                self.processed_image.save(path)
                self.show_toast("Image Saved!")

    def rotate_image(self, angle):
        if self.original_image:
            self.processed_image = self.processed_image.rotate(angle, expand=True)
            self.update_display()
            self.show_toast(f"Rotated {angle}°")

    def flip_image(self):
        if self.original_image:
            self.processed_image = ImageOps.mirror(self.processed_image)
            self.update_display()
            self.show_toast("Image Flipped")

    def show_toast(self, message, is_error=False):
        color = "#ef4444" if is_error else "#10b981"
        self.toast.configure(text=message, fg_color=color)
        self.toast.place(rely=0.08)
        self.after(2000, lambda: self.toast.place(rely=-0.1))

    def on_closing(self):
        if self.original_image:
            if messagebox.askokcancel("Quit", "You have an image open. Unsaved changes will be lost.\nDo you really want to quit?"):
                self.destroy()
        else:
            self.destroy()

    def _on_resize(self, event):
        if self.editor_frame.winfo_ismapped():
            if hasattr(self, '_resize_job'): self.after_cancel(self._resize_job)
            self._resize_job = self.after(100, self.update_display)

if __name__ == "__main__":
    app = ModernFilterApp()
    app.mainloop()