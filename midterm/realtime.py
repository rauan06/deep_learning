#!/usr/bin/env python3
"""
Real-Time Digit Recognition using  DigitRecognizer class
This imports and uses  trained model directly!
"""

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import cv2
import time
import sys
import os

# Import  DigitRecognizer class
sys.path.insert(0, os.getcwd())  # Add current directory to path
sys.path.append('/home/claude')
sys.path.append('/mnt/user-data/outputs')

try:
    from digit_recognizer import DigitRecognizer
    print("Successfully imported DigitRecognizer class!")
except ImportError as e:
    print(f"Could not import DigitRecognizer: {e}")
    print("Make sure digit_recognizer.py is in the same directory!")


class RealTimeDigitGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Digit Recognition - Using  Model")
        self.root.geometry("900x600")
        self.root.configure(bg='#2c3e50')
        
        #  DigitRecognizer instance
        self.recognizer = DigitRecognizer()
        self.model_loaded = False
        
        # Drawing variables
        self.canvas_size = 400
        self.drawing = False
        self.last_x = None
        self.last_y = None
        self.brush_size = 20
        
        # PIL Image for drawing
        self.image = Image.new('RGB', (self.canvas_size, self.canvas_size), 'white')
        self.draw = ImageDraw.Draw(self.image)
        
        # Real-time prediction settings
        self.auto_predict = True
        self.prediction_delay = 500
        self.last_draw_time = 0
        self.prediction_scheduled = False
        
        self._setup_ui()
        self._load__model()
        
    def _setup_ui(self):
        """Setup the user interface."""
        
        # Main container
        main_container = tk.Frame(self.root, bg='#2c3e50')
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header = tk.Frame(main_container, bg='#3498db', height=90)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(
            header,
            text="Real-Time Digit Recognition",
            font=('Arial', 28, 'bold'),
            bg='#3498db',
            fg='white'
        ).pack(pady=(15, 5))
        
        tk.Label(
            header,
            text="Using DigitRecognizer Model",
            font=('Arial', 12, 'italic'),
            bg='#3498db',
            fg='#ecf0f1'
        ).pack()
        
        # Content area
        content = tk.Frame(main_container, bg='#2c3e50')
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left side - Drawing area
        left_frame = tk.Frame(content, bg='#34495e', relief=tk.RAISED, bd=3)
        left_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Label(
            left_frame,
            text="Draw Here (predicts automatically!)",
            font=('Arial', 14, 'bold'),
            bg='#34495e',
            fg='white'
        ).pack(pady=10)
        
        # Canvas
        self.canvas = tk.Canvas(
            left_frame,
            width=self.canvas_size,
            height=self.canvas_size,
            bg='white',
            cursor='crosshair',
            highlightthickness=0
        )
        self.canvas.pack(padx=10, pady=10)
        
        # Bind mouse events
        self.canvas.bind('<Button-1>', self._on_mouse_press)
        self.canvas.bind('<B1-Motion>', self._on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_mouse_release)
        
        # Controls
        control_frame = tk.Frame(left_frame, bg='#34495e')
        control_frame.pack(pady=10)
        
        # Brush size
        tk.Label(
            control_frame,
            text="Brush Size:",
            font=('Arial', 10),
            bg='#34495e',
            fg='white'
        ).grid(row=0, column=0, padx=5)
        
        self.brush_scale = tk.Scale(
            control_frame,
            from_=10,
            to=40,
            orient=tk.HORIZONTAL,
            length=150,
            command=self._update_brush_size,
            bg='#34495e',
            fg='white',
            highlightthickness=0
        )
        self.brush_scale.set(self.brush_size)
        self.brush_scale.grid(row=0, column=1, padx=5)
        
        # Clear button
        clear_btn = tk.Button(
            left_frame,
            text="CLEAR",
            command=self._clear_canvas,
            font=('Arial', 12, 'bold'),
            bg='#e74c3c',
            fg='white',
            width=15,
            height=2,
            cursor='hand2',
            relief=tk.RAISED,
            bd=3
        )
        clear_btn.pack(pady=10)
        
        # Right side - Prediction display
        right_frame = tk.Frame(content, bg='#34495e', relief=tk.RAISED, bd=3)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        tk.Label(
            right_frame,
            text="PREDICTION",
            font=('Arial', 16, 'bold'),
            bg='#34495e',
            fg='white'
        ).pack(pady=15)
        
        # Large digit display
        digit_frame = tk.Frame(right_frame, bg='#2c3e50', relief=tk.SUNKEN, bd=5)
        digit_frame.pack(pady=10, padx=20)
        
        self.prediction_label = tk.Label(
            digit_frame,
            text="?",
            font=('Arial', 120, 'bold'),
            bg='#2c3e50',
            fg='#95a5a6',
            width=3,
            height=1
        )
        self.prediction_label.pack(padx=40, pady=40)
        
        # Confidence
        self.confidence_label = tk.Label(
            right_frame,
            text="Confidence: --%",
            font=('Arial', 14, 'bold'),
            bg='#34495e',
            fg='#ecf0f1'
        )
        self.confidence_label.pack(pady=5)
        
        # Status indicator
        self.status_label = tk.Label(
            right_frame,
            text="● Loading model...",
            font=('Arial', 12),
            bg='#34495e',
            fg='#f39c12'
        )
        self.status_label.pack(pady=5)
        
        # Separator
        tk.Frame(right_frame, bg='#7f8c8d', height=2).pack(fill=tk.X, pady=15, padx=20)
        
        # All probabilities
        tk.Label(
            right_frame,
            text="ALL PROBABILITIES",
            font=('Arial', 12, 'bold'),
            bg='#34495e',
            fg='white'
        ).pack(pady=10)
        
        # Probability bars
        prob_container = tk.Frame(right_frame, bg='#34495e')
        prob_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        self.prob_bars = []
        self.prob_labels = []
        
        for i in range(10):
            row_frame = tk.Frame(prob_container, bg='#34495e')
            row_frame.pack(fill=tk.X, pady=3)
            
            # Digit label
            tk.Label(
                row_frame,
                text=f"{i}:",
                font=('Arial', 11, 'bold'),
                bg='#34495e',
                fg='white',
                width=2
            ).pack(side=tk.LEFT, padx=5)
            
            # Progress bar background
            bar_bg = tk.Canvas(row_frame, width=250, height=20, bg='#2c3e50', highlightthickness=0)
            bar_bg.pack(side=tk.LEFT, padx=5)
            
            # Progress bar
            bar = bar_bg.create_rectangle(0, 0, 0, 20, fill='#3498db', outline='')
            self.prob_bars.append((bar_bg, bar))
            
            # Percentage label
            pct_label = tk.Label(
                row_frame,
                text="0%",
                font=('Arial', 10),
                bg='#34495e',
                fg='#ecf0f1',
                width=6,
                anchor='w'
            )
            pct_label.pack(side=tk.LEFT, padx=5)
            self.prob_labels.append(pct_label)
        
        # Footer
        footer = tk.Frame(main_container, bg='#34495e', height=50)
        footer.pack(fill=tk.X)
        footer.pack_propagate(False)
        
        self.auto_predict_var = tk.BooleanVar(value=True)
        auto_check = tk.Checkbutton(
            footer,
            text="Auto-Predict (Real-Time)",
            variable=self.auto_predict_var,
            command=self._toggle_auto_predict,
            font=('Arial', 11, 'bold'),
            bg='#34495e',
            fg='white',
            selectcolor='#2c3e50',
            activebackground='#34495e',
            activeforeground='white'
        )
        auto_check.pack(pady=12)
        
    def _update_brush_size(self, value):
        """Update brush size."""
        self.brush_size = int(float(value))
        
    def _toggle_auto_predict(self):
        """Toggle auto-predict mode."""
        self.auto_predict = self.auto_predict_var.get()
        if self.auto_predict:
            self.status_label.config(text="● Auto-Predict ON", fg='#2ecc71')
        else:
            self.status_label.config(text="○ Auto-Predict OFF", fg='#95a5a6')
    
    def _on_mouse_press(self, event):
        """Handle mouse press."""
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y
        self.last_draw_time = time.time()
        
    def _on_mouse_drag(self, event):
        """Handle mouse drag."""
        if self.drawing:
            # Draw on canvas
            self.canvas.create_line(
                self.last_x, self.last_y, event.x, event.y,
                width=self.brush_size,
                fill='black',
                capstyle=tk.ROUND,
                smooth=True
            )
            
            # Draw on PIL image
            self.draw.line(
                [self.last_x, self.last_y, event.x, event.y],
                fill='black',
                width=self.brush_size
            )
            
            self.last_x = event.x
            self.last_y = event.y
            self.last_draw_time = time.time()
            
            # Schedule prediction
            if self.auto_predict and not self.prediction_scheduled:
                self.prediction_scheduled = True
                self.root.after(self.prediction_delay, self._auto_predict)
    
    def _on_mouse_release(self, event):
        """Handle mouse release."""
        self.drawing = False
        
    def _auto_predict(self):
        """Automatically predict after drawing stops."""
        self.prediction_scheduled = False
        
        # Check if enough time has passed
        time_since_draw = (time.time() - self.last_draw_time) * 1000
        
        if time_since_draw >= self.prediction_delay:
            self._predict()
        else:
            # Reschedule
            remaining_time = int(self.prediction_delay - time_since_draw)
            self.prediction_scheduled = True
            self.root.after(remaining_time, self._auto_predict)
    
    def _predict(self):
        """Make prediction using DigitRecognizer."""
        if not self.model_loaded:
            return
        
        # Check if canvas is empty
        if self.image.getbbox() is None:
            return
        
        try:
            # Preprocess image (same as  recognizer expects)
            processed_image = self._preprocess_image()
            
            if processed_image is None:
                return
            
            # Use  recognizer's predict_digit method!
            digit, confidence = self.recognizer.predict_digit(processed_image)
            
            # Also get full prediction probabilities
            prediction = self.recognizer.model.predict(processed_image, verbose=0)
            
            # Update display
            color = '#2ecc71' if confidence > 90 else '#f39c12' if confidence > 70 else '#e74c3c'
            self.prediction_label.config(text=str(digit), fg=color)
            self.confidence_label.config(text=f"Confidence: {confidence:.1f}%")
            
            # Update probability bars
            for i in range(10):
                prob = prediction[0][i] * 100
                bar_bg, bar = self.prob_bars[i]
                
                # Update bar width
                bar_width = int((prob / 100) * 250)
                bar_bg.coords(bar, 0, 0, bar_width, 20)
                
                # Color based on probability
                if i == digit:
                    bar_bg.itemconfig(bar, fill='#2ecc71')
                else:
                    bar_bg.itemconfig(bar, fill='#3498db')
                
                # Update label
                self.prob_labels[i].config(text=f"{prob:.1f}%")
            
        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
    
    def _preprocess_image(self):
        """Preprocess image (same format as  model expects)."""
        try:
            # Convert to grayscale
            img_gray = ImageOps.grayscale(self.image)
            
            # Invert
            img_inverted = ImageOps.invert(img_gray)
            
            # Convert to numpy
            img_array = np.array(img_inverted)
            
            # Find bounding box
            coords = cv2.findNonZero(img_array)
            
            if coords is None:
                return None
            
            x, y, w, h = cv2.boundingRect(coords)
            
            # Add padding
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(self.canvas_size - x, w + 2 * padding)
            h = min(self.canvas_size - y, h + 2 * padding)
            
            # Crop
            img_cropped = img_array[y:y+h, x:x+w]
            
            # Make square
            if w > h:
                diff = w - h
                top_pad = diff // 2
                bottom_pad = diff - top_pad
                img_cropped = np.pad(img_cropped, ((top_pad, bottom_pad), (0, 0)), mode='constant')
            elif h > w:
                diff = h - w
                left_pad = diff // 2
                right_pad = diff - left_pad
                img_cropped = np.pad(img_cropped, ((0, 0), (left_pad, right_pad)), mode='constant')
            
            # Resize to 28x28 (what  model expects!)
            img_resized = cv2.resize(img_cropped, (28, 28), interpolation=cv2.INTER_AREA)
            
            # Normalize (0-1 range, like  model expects!)
            img_normalized = img_resized.astype('float32') / 255.0
            
            # Reshape for  model (1, 28, 28, 1)
            img_final = img_normalized.reshape(1, 28, 28, 1)
            
            return img_final
            
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None
    
    def _clear_canvas(self):
        """Clear the canvas."""
        self.canvas.delete('all')
        self.image = Image.new('RGB', (self.canvas_size, self.canvas_size), 'white')
        self.draw = ImageDraw.Draw(self.image)
        
        # Reset display
        self.prediction_label.config(text="?", fg='#95a5a6')
        self.confidence_label.config(text="Confidence: --%")
        
        # Reset bars
        for i in range(10):
            bar_bg, bar = self.prob_bars[i]
            bar_bg.coords(bar, 0, 0, 0, 20)
            self.prob_labels[i].config(text="0%")
    
    def _load__model(self):
        """Load  trained model using  DigitRecognizer class."""
        try:
            # Use  recognizer's load_model method!
            # Try current directory first
            model_path = 'digit_recognizer_model.h5'
            
            if not os.path.exists(model_path):
                # Try outputs directory as fallback
                model_path = '/mnt/user-data/outputs/digit_recognizer_model.h5'
            
            print(f"Loading  model from: {model_path}")
            self.recognizer.load_model(model_path)
            self.model_loaded = True
            
            self.status_label.config(text="● Model Loaded!", fg='#2ecc71')
            print("Successfully loaded trained model!")
            
        except FileNotFoundError:
            self.status_label.config(text="✗ Model not found - Train first!", fg='#e74c3c')
            messagebox.showerror(
                "Model Not Found",
                " model file not found!\n\n"
                "Please train  model first by running:\n"
                "python digit_recognizer.py"
            )
        except Exception as e:
            self.status_label.config(text="✗ Error loading model", fg='#e74c3c')
            messagebox.showerror("Error", f"Failed to load  model:\n{e}")
            import traceback
            traceback.print_exc()


def main():
    """Run the real-time GUI with  DigitRecognizer."""
    print("="*60)
    print("Real-Time Digit Recognition")
    print("")
    print("="*60)
    
    root = tk.Tk()
    app = RealTimeDigitGUI(root)
    
    # Center window
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    root.mainloop()


if __name__ == "__main__":
    main()