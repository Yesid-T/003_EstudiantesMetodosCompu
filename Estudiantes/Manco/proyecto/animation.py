import numpy as np
import pandas as pd
from vispy import app, gloo
from vispy.util.transforms import ortho
import os
import cv2
from datetime import datetime

vertex_shader = """
uniform mat4 u_projection;
uniform float u_point_size;
attribute vec2 a_position;
attribute float a_size;
attribute float a_opacity;
varying float v_opacity;
void main() {
    gl_Position = u_projection * vec4(a_position, 0.0, 1.0);
    gl_PointSize = a_size;
    v_opacity = a_opacity;
}
"""

fragment_shader = """
uniform vec4 u_color;
varying float v_opacity;
void main() {
    float r = length(gl_PointCoord - vec2(0.5));
    float glow = smoothstep(0.3, 0.0, r);
    float neon = smoothstep(0.2, 0.0, r);
    vec3 neonColor = u_color.rgb * (1.0 + 0.8 * neon);
    gl_FragColor = vec4(neonColor, glow * u_color.a * v_opacity + neon * 0.7 * v_opacity);
}
"""

class OrbitCanvas(app.Canvas):
    def __init__(self, data_file):
        # Resolución estándar cuadrada para videos (1080x1080)
        self.video_resolution = (1080, 1080)
        
        app.Canvas.__init__(self, title='Problema de Tres Cuerpos', size=(1200, 1200), keys='interactive')

        # Cargar datos del CSV
        self.load_data(data_file)
        
        self.program = gloo.Program(vertex_shader, fragment_shader)
        
        # Configuración del trail
        self.trail_length = 800
        self.trail_positions = np.zeros((3, self.trail_length, 2), dtype=np.float32)
        self.trail_opacities = np.linspace(0.5, 0.0, self.trail_length)
        self.trail_sizes = np.linspace(20.0, 15.0, self.trail_length)

        # Crear buffers para las trails
        self.trail_vbos = []
        self.trail_data = []
        
        # Crear un VBO para cada cuerpo con formato de datos
        for i in range(3):
            # Inicializar datos para el trail
            trail_data = np.zeros(self.trail_length, 
                                 [('a_position', np.float32, 2),
                                  ('a_size', np.float32),
                                  ('a_opacity', np.float32)])
            vbo = gloo.VertexBuffer(trail_data)
            self.trail_vbos.append(vbo)
            self.trail_data.append(trail_data)
        
        # Crear VBOs para las partículas principales (optimización)
        self.main_vbos = []
        self.main_data = []
        for i in range(3):
            main_data = np.zeros(1, [('a_position', np.float32, 2),
                                    ('a_size', np.float32),
                                    ('a_opacity', np.float32)])
            main_data['a_position'][0] = (0, 0)  # Se actualizará luego
            main_data['a_size'][0] = 180.0
            main_data['a_opacity'][0] = 1.0
            self.main_vbos.append(gloo.VertexBuffer(main_data))
            self.main_data.append(main_data)
        
        self.program['u_color'] = (1.0, 1.0, 1.0, 1.0)
        self.program['u_point_size'] = 180.0
        
        # Índice para recorrer los datos
        self.data_index = 0
        self.positions = np.zeros((3, 2), dtype=np.float32)
        
        # Control de velocidad - número de frames por actualización
        self.speed = 60
        
        # Variables para grabación de video
        self.recording = True  # Comenzar grabando automáticamente
        self.video_writer = None
        self.video_frames = []
        self.video_fps = 60  # Frames por segundo del video
        
        # Ruta donde se guardarán los videos
        self.video_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos")
        
        # Cálculo de la duración basada en los parámetros de simulación
        # Con dt=0.0005 y steps=50000, el tiempo total de simulación es 0.0005 * 50000 = 25 segundos
        self.total_simulation_time = 0.0005 * 50000 / 2  # 25 segundos
        
        # Ajustamos la duración del video para que coincida con la simulación
        self.duration = self.total_simulation_time  # 25 segundos
        
        self.frame_count = 0
        self.total_frames_to_record = self.video_fps * self.duration
        
        # Ajustar la velocidad de reproducción para que toda la simulación se vea en el tiempo correcto
        self.playback_speed = self.total_frames / (self.video_fps * self.duration)
        self.speed = int(self.playback_speed)
        
        print("-" * 60)
        print(f"CONFIGURACIÓN DE LA SIMULACIÓN:")
        print(f"- Tiempo total de simulación: {self.total_simulation_time} segundos")
        print(f"- Duración del video: {self.duration} segundos")
        print(f"- Velocidad de reproducción: {self.speed} frames por step")
        print(f"- Resolución de ventana: {self.size[0]}x{self.size[1]}")
        print(f"- Resolución de video estándar: {self.video_resolution[0]}x{self.video_resolution[1]} (formato cuadrado)")
        print(f"- Calidad de video: {self.video_fps} FPS")
        print(f"- Grabación automática: ACTIVADA")
        print(f"- Ubicación del video: {self.video_dir}")
        print("-" * 60)
        
        # Inicializar posiciones
        self.update_positions()

        self.timer = app.Timer('auto', connect=self.on_timer, start=True)
        gloo.set_state(clear_color='black', blend=True, blend_func=('src_alpha', 'one'))
        
        # Registrar evento de teclado
        self.connect(self.on_key_press)
        
        print("Controles:")
        print("  + : Aumentar velocidad")
        print("  - : Disminuir velocidad")
        print("  r : Restablecer velocidad")
        print("  v : Iniciar/Detener grabación de video (20 segundos)")

        # Inicializar la proyección para que sea visible toda la animación
        self.program['u_projection'] = ortho(-6, 6, -6, 6, -1, 1)

    def load_data(self, data_file):
        """Carga los datos del CSV"""
        self.df = pd.read_csv(data_file)
        
        # Establecer los límites fijos para normalizar las posiciones
        self.x_min = -5
        self.x_max = 5
        self.y_min = -5
        self.y_max = 5
        
        # Calcular el rango para escalar
        self.x_range = self.x_max - self.x_min
        self.y_range = self.y_max - self.y_min
        
        # Determinar el número total de frames
        self.total_frames = len(self.df)
        print(f"Datos cargados: {self.total_frames} frames")

    def normalize_position(self, x, y):
        """Normaliza las coordenadas al rango [-5, 5]"""
        x_norm = -5.0 + 10.0 * (x - self.x_min) / self.x_range
        y_norm = -5.0 + 10.0 * (y - self.y_min) / self.y_range
        return x_norm, y_norm

    def on_key_press(self, event):
        """Maneja eventos de teclado para controlar la velocidad"""
        if event.key == '+':
            self.speed = min(20, self.speed + 1)  # Limitar a 20x
            print(f"Velocidad: {self.speed}x")
        elif event.key == '-':
            self.speed = max(1, self.speed - 1)  # Mínimo 1x
            print(f"Velocidad: {self.speed}x")
        elif event.key == 'r':
            self.speed = 1
            print("Velocidad restablecida: 1x")
        elif event.key == 'v':
            self.toggle_recording()

    def toggle_recording(self):
        """Inicia o detiene la grabación del video"""
        if not self.recording:
            # Iniciar grabación
            self.recording = True
            self.frame_count = 0
            self.video_frames = []
            self.data_index = 0  # Reiniciar la simulación al inicio
            print(f"Iniciando grabación de video de {self.duration} segundos...")
        else:
            # Finalizar grabación (si estaba en progreso)
            self.recording = False
            if self.frame_count > 0:
                self.save_video()
            print("Grabación detenida.")

    def capture_frame(self):
        """Captura el frame actual para el video"""
        if self.recording and self.frame_count < self.total_frames_to_record:
            # Capturar el buffer de la pantalla
            img = gloo.read_pixels()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convertir de RGB a BGR para OpenCV
            
            # Redimensionar inmediatamente a la resolución estándar si es necesario
            if img.shape[0] != self.video_resolution[1] or img.shape[1] != self.video_resolution[0]:
                img = cv2.resize(img, self.video_resolution, interpolation=cv2.INTER_LANCZOS4)
                
            self.video_frames.append(img)
            self.frame_count += 1
            
            # Mostrar progreso
            if self.frame_count % 30 == 0:
                progress = (self.frame_count / self.total_frames_to_record) * 100
                print(f"Grabando: {progress:.1f}% completado")
            
            # Si completamos todos los frames, guardar el video
            if self.frame_count >= self.total_frames_to_record:
                self.save_video()
                self.recording = False
                print(f"Grabación completada. El video se ha guardado en: {self.video_dir}")

    def save_video(self):
        """Guarda los frames capturados como un archivo de video"""
        if not self.video_frames:
            print("No hay frames para guardar.")
            return
        
        # Crear directorio de videos dentro del proyecto si no existe
        os.makedirs(self.video_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"three_body_animation_{timestamp}.mp4"
        filepath = os.path.join(self.video_dir, filename)
        
        # Configurar el codificador de video con mayor calidad y resolución estándar
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Crear el video con la resolución estándar
        out = cv2.VideoWriter(filepath, fourcc, self.video_fps, self.video_resolution, isColor=True)
        
        # Escribir cada frame al video con la máxima calidad
        for frame in self.video_frames:
            out.write(frame)
        
        # Finalizar y liberar
        out.release()
        print("\n" + "="*60)
        print(f"¡VIDEO GUARDADO EXITOSAMENTE!")
        print(f"Ruta completa: {os.path.abspath(filepath)}")
        print(f"Carpeta: {self.video_dir}")
        print(f"Archivo: {filename}")
        print(f"Resolución: {self.video_resolution[0]}x{self.video_resolution[1]}, FPS: {self.video_fps}")
        print("="*60 + "\n")
        self.video_frames = []  # Liberar memoria

    def update_positions(self):
        """Actualiza las posiciones con los datos del CSV"""
        if self.data_index < self.total_frames:
            row = self.df.iloc[self.data_index]
            
            # Normalizar las posiciones a nuestro sistema de coordenadas
            x1, y1 = self.normalize_position(row['x1'], row['y1'])
            x2, y2 = self.normalize_position(row['x2'], row['y2'])
            x3, y3 = self.normalize_position(row['x3'], row['y3'])
            
            # Actualizar las posiciones
            self.positions[0] = [x1, y1]
            self.positions[1] = [x2, y2]
            self.positions[2] = [x3, y3]
            
            # Avanzar al siguiente frame según la velocidad
            self.data_index = (self.data_index + self.speed) % self.total_frames

    def on_resize(self, event):
        w, h = event.size
        gloo.set_viewport(0, 0, w, h)
        # Ampliar el rango de la proyección ortográfica para ver toda la animación
        self.program['u_projection'] = ortho(-6, 6, -6, 6, -1, 1)

    def on_draw(self, event):
        gloo.clear()
        
        for i, color in enumerate([
            (0.0, 1.0, 1.0, 1.0),    # Cyan neón brillante
            (1.0, 0.0, 0.5, 1.0),    # Rosa neón
            (0.0, 1.0, 0.0, 1.0)     # Verde neón
        ]):
            # Actualizar datos de la trail reutilizando el array pre-creado
            for j in range(self.trail_length):
                if np.any(self.trail_positions[i, j] != 0):
                    self.trail_data[i]['a_position'][j] = self.trail_positions[i, j]
                    self.trail_data[i]['a_size'][j] = self.trail_sizes[j]
                    self.trail_data[i]['a_opacity'][j] = self.trail_opacities[j]
            
            # Actualizar VBO y dibujar el trail completo
            self.trail_vbos[i].set_data(self.trail_data[i])
            self.program.bind(self.trail_vbos[i])
            self.program['u_color'] = color
            self.program.draw('points')
            
            # Actualizar la posición de la partícula principal
            self.main_data[i]['a_position'][0] = self.positions[i]
            
            # Actualizar el VBO de la partícula principal
            self.main_vbos[i].set_data(self.main_data[i])
            
            # Actualizar el shader y dibujar la partícula principal
            self.program.bind(self.main_vbos[i])
            self.program['u_color'] = color
            self.program.draw('points')
            
        # Capturar el frame para el video si estamos grabando
        if self.recording:
            self.capture_frame()

    def on_timer(self, event):
        # Actualizar trail (desplazar viejas posiciones y añadir la nueva)
        self.trail_positions = np.roll(self.trail_positions, 1, axis=1)
        self.trail_positions[:, 0] = self.positions
        
        # Actualizar posiciones desde los datos
        self.update_positions()
        
        self.update()


if __name__ == '__main__':
    data_file = 'three_body_solution_2_.csv'
    canvas = OrbitCanvas(data_file)
    canvas.show()
    app.run()
