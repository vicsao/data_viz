import numpy as np
import cv2
import librosa
import os
from moviepy.editor import VideoClip, AudioFileClip, concatenate_audioclips

# --- SETTINGS ---
# Point this to the FOLDER containing your tracks
AUDIO_FOLDER = r"E:\PYTHON\SYNTHCREATIONZ\test_assets\synthwave"
DEV_MODE = True    
DEV_LIMIT = 70     
FADE_FACTOR = 0.95 
MAX_FLICK_DIST = 220 
MS_HEALTH = 15     
RESET_DELAY = 0.8  
END_TRANSITION_TIME = 10.0 

class SupernovaGrinder:
    def __init__(self, res, onset_env, times, total_duration):
        self.res = res
        self.onset_env = onset_env
        self.times = times
        self.total_duration = total_duration
        self.canvas = np.zeros((res[1], res[0], 3), dtype=np.uint8)
        self.score_yellow = 0
        self.score_blue = 0
        self.particles = [] 
        self.reset_game()

    def reset_game(self):
        self.base_yellow = (150, self.res[1]//2)
        self.base_blue = (self.res[0]-150, self.res[1]//2)
        self.fleet_yellow = [self.base_yellow]
        self.fleet_blue = [self.base_blue]
        self.hp_yellow, self.hp_blue = MS_HEALTH, MS_HEALTH
        self.last_turn_id = -1
        self.has_flicked_this_turn = False
        self.game_over = False
        self.winner_time = None
        self.winner_name = None
        self.final_explosion_triggered = False
        self.grudge_yellow = None 
        self.grudge_blue = None   

    def check_collision(self, p1, p2, target_list, is_ms_check=False):
        p1, p2 = np.array(p1), np.array(p2)
        hit_indices = []
        for i, ship in enumerate(target_list):
            ship_p = np.array(ship)
            line_vec = p2 - p1
            line_len = np.linalg.norm(line_vec)
            if line_len == 0: continue
            line_unit = line_vec / line_len
            ship_vec = ship_p - p1
            projection = np.dot(ship_vec, line_unit)
            if 0 <= projection <= line_len:
                closest_p = p1 + line_unit * projection
                hit_radius = 25 if (is_ms_check and i == 0) else 10
                if np.linalg.norm(ship_p - closest_p) < hit_radius:
                    hit_indices.append(i)
        return hit_indices

    def process_frame(self, t):
        time_left = self.total_duration - t
        if time_left < END_TRANSITION_TIME and not self.final_explosion_triggered:
            self.trigger_pixel_explosion()
            self.game_over = True

        self.canvas = (self.canvas * FADE_FACTOR).astype(np.uint8)
        for p in self.particles[:]:
            p[0] += p[2]; p[1] += p[3]; p[5] -= 0.015
            if p[5] <= 0: self.particles.remove(p)
            else: cv2.circle(self.canvas, (int(p[0]), int(p[1])), 1, p[4], -1)

        cv2.putText(self.canvas, f"YELLOW: {self.score_yellow}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(self.canvas, f"BLUE: {self.score_blue}", (self.res[0]-250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

        if self.game_over and not self.final_explosion_triggered:
            if self.winner_time is None: 
                self.winner_time = t
                if self.winner_name == "YELLOW": self.score_yellow += 1
                else: self.score_blue += 1
            f = self.canvas.copy()
            cv2.putText(f, f"{self.winner_name} VICTORIOUS", (self.res[0]//2-300, self.res[1]//2), 2, 2, (255, 255, 255), 4)
            if t - self.winner_time > RESET_DELAY:
                self.reset_game()
                self.canvas = np.zeros((self.res[1], self.res[0], 3), dtype=np.uint8)
            return f

        if time_left < END_TRANSITION_TIME:
            final_winner = "YELLOW" if self.score_yellow > self.score_blue else "BLUE"
            if self.score_yellow == self.score_blue: final_winner = "DRAW"
            cv2.putText(self.canvas, f"GRAND CHAMPION: {final_winner}", (self.res[0]//2-450, self.res[1]//2), 2, 2, (255, 255, 255), 4)
            return self.canvas.copy()

        idx = np.searchsorted(self.times, t)
        power = self.onset_env[idx] if idx < len(self.onset_env) else 0
        turn_rate = 2 + (power * 1.0)
        turn_id = int(t * turn_rate)
        if turn_id != self.last_turn_id:
            self.has_flicked_this_turn = False; self.last_turn_id = turn_id

        if power > 0.9 and not self.has_flicked_this_turn:
            team = "YELLOW" if turn_id % 2 == 0 else "BLUE"
            color = (255, 255, 0) if team == "YELLOW" else (0, 165, 255)
            my_fleet, enemies = (self.fleet_yellow, self.fleet_blue) if team == "YELLOW" else (self.fleet_blue, self.fleet_yellow)
            my_grudge = self.grudge_yellow if team == "YELLOW" else self.grudge_blue
            is_base_firing = False
            
            if my_grudge and any(np.array_equal(my_grudge, e) for e in enemies):
                target_pos, start_pt, is_base_firing = my_grudge, my_fleet[0], True
            else:
                mid_x = self.res[0] // 2
                threats = [p for p in enemies if (team == "YELLOW" and p[0] < mid_x) or (team == "BLUE" and p[0] > mid_x)]
                if threats:
                    dist_to_base = [np.linalg.norm(np.array(p) - np.array(my_fleet[0])) for p in threats]
                    closest_idx = np.argmin(dist_to_base)
                    if dist_to_base[closest_idx] < 300:
                        start_pt, target_pos, is_base_firing = my_fleet[0], threats[closest_idx], True
                    else:
                        target_pos = threats[closest_idx]
                        start_pt = my_fleet[np.argmin([np.linalg.norm(np.array(p)-np.array(target_pos)) for p in my_fleet])]
                else:
                    target_pos, start_pt = enemies[0], my_fleet[np.argmax([p[0] if team == "YELLOW" else -p[0] for p in my_fleet])]

            # --- DYNAMIC RANGE (1.5x Multiplier) ---
            angle = np.arctan2(target_pos[1]-start_pt[1], target_pos[0]-start_pt[0]) + np.deg2rad(np.random.uniform(-2.5 if is_base_firing else -15, 2.5 if is_base_firing else 15))
            
            base_dist = 80 + (power * 60)
            range_mult = 1.5 if is_base_firing else 1.0
            dist = np.clip(base_dist * range_mult, 50, MAX_FLICK_DIST * range_mult)
            
            end_pt = (int(start_pt[0] + np.cos(angle)*dist), int(start_pt[1] + np.sin(angle)*dist))
            end_pt = (np.clip(end_pt[0], 0, self.res[0]), np.clip(end_pt[1], 0, self.res[1]))

            # --- RESOLVE (Spawn on Miss) ---
            hit_detected = False
            if self.check_collision(start_pt, end_pt, enemies[:1], True):
                hit_detected = True
                if team == "YELLOW": 
                    self.hp_blue -= 1; self.grudge_blue = start_pt 
                else: 
                    self.hp_yellow -= 1; self.grudge_yellow = start_pt 
                cv2.circle(self.canvas, enemies[0], 45, (255, 255, 255), 3)
                if self.hp_yellow <= 0 or self.hp_blue <= 0:
                    self.game_over = True; self.winner_name = "YELLOW" if self.hp_blue <= 0 else "BLUE"

            f_kills = self.check_collision(start_pt, end_pt, enemies)
            for k_idx in sorted(f_kills, reverse=True):
                if k_idx != 0: 
                    hit_detected = True
                    death_pt = enemies.pop(k_idx); cv2.circle(self.canvas, death_pt, 15, (255, 255, 255), 2)
                    if team == "YELLOW" and np.array_equal(death_pt, self.grudge_yellow): self.grudge_yellow = None
                    if team == "BLUE" and np.array_equal(death_pt, self.grudge_blue): self.grudge_blue = None

            if not hit_detected: my_fleet.append(end_pt)
            # --- DRAW LASER (Dynamic Thickness) ---
            # Use 2 for Mother Ship (Heavy Beam), 1 for Fighters (Light Beam)
            laser_thickness = 2 if is_base_firing else 1
            cv2.line(self.canvas, start_pt, end_pt, color, laser_thickness, cv2.LINE_AA)
            
            if not hit_detected: my_fleet.append(end_pt)
            self.has_flicked_this_turn = True

        for p in self.fleet_yellow: cv2.circle(self.canvas, p, 3, (255, 255, 0), -1)
        for p in self.fleet_blue: cv2.circle(self.canvas, p, 3, (0, 165, 255), -1)
        cv2.circle(self.canvas, self.base_yellow, 20 + self.hp_yellow, (255, 255, 0), 2)
        cv2.circle(self.canvas, self.base_blue, 20 + self.hp_blue, (0, 165, 255), 2)
        return self.canvas.copy()

    def trigger_pixel_explosion(self):
        all_units = [(p, (255, 255, 0)) for p in self.fleet_yellow] + [(p, (0, 165, 255)) for p in self.fleet_blue]
        for pos, color in all_units:
            for _ in range(12): 
                vx, vy = np.random.uniform(-7, 7), np.random.uniform(-7, 7)
                self.particles.append([pos[0], pos[1], vx, vy, color, 1.0])
        self.fleet_yellow, self.fleet_blue = [], []; self.final_explosion_triggered = True

# --- DYNAMIC AUDIO LOADER ---
print(f"📂 Scanning folder: {AUDIO_FOLDER}")
valid_exts = ('.wav', '.mp3')
audio_files = [os.path.join(AUDIO_FOLDER, f) for f in os.listdir(AUDIO_FOLDER) if f.lower().endswith(valid_exts)]

if not audio_files:
    raise FileNotFoundError("❌ No .wav or .mp3 files found in the specified folder!")

clips = [AudioFileClip(f) for f in audio_files]
final_audio = concatenate_audioclips(clips)

# Export a temp mix for Librosa to analyze
TEMP_MIX = "temp_mix.wav"
final_audio.write_audiofile(TEMP_MIX, fps=22050, verbose=False, logger=None)

# --- RUNNER ---
dur = min(final_audio.duration, DEV_LIMIT) if DEV_MODE else final_audio.duration
y, sr = librosa.load(TEMP_MIX, sr=22050, duration=dur)
onset_env = librosa.onset.onset_strength(y=y, sr=sr)
times = librosa.times_like(onset_env, sr=sr)

war = SupernovaGrinder((1280, 720), onset_env, times, dur)
video = VideoClip(war.process_frame, duration=dur).set_audio(final_audio.subclip(0, dur))
video.write_videofile("Starfighter_MegaMix.mp4", fps=60, codec="h264_nvenc", ffmpeg_params=["-pix_fmt", "yuv420p"])

# Cleanup
if os.path.exists(TEMP_MIX): os.remove(TEMP_MIX)