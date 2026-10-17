import cv2
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN, PPO

# 1. Initialize two environments using 'rgb_array' render mode
env_dqn = gym.make("LunarLander-v3", render_mode="rgb_array")
env_ppo = gym.make("LunarLander-v3", render_mode="rgb_array")

# 2. Load trained models
print("Loading trained models...")
model_dqn = DQN.load("dqn_lunarlander")
model_ppo = PPO.load("ppo_lunarlander")

# 3. Reset both environments with the same seed for a fair comparison
obs_dqn, _ = env_dqn.reset(seed=42)
obs_ppo, _ = env_ppo.reset(seed=42)

sample_frame_dqn = env_dqn.render()
sample_frame_ppo = env_ppo.render()

height, width, _ = sample_frame_dqn.shape
combined_width = width * 2

output_fillname = "lunar_lander_comparison.mp4"
fps = 50.0
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_fillname, fourcc, fps, (combined_width, height))

max_episodes = 5
completed_episodes = 0

print(
    "Displaying side-by-side comparison! Press 'q' or 'Esc' in the render window to exit."
)

while True:
    # Get model action predictions
    action_dqn, _ = model_dqn.predict(obs_dqn, deterministic=True)
    action_ppo, _ = model_ppo.predict(obs_ppo, deterministic=True)

    # Step simulation forward
    obs_dqn, _, term_dqn, trunc_dqn, _ = env_dqn.step(action_dqn)
    obs_ppo, _, term_ppo, trunc_ppo, _ = env_ppo.step(action_ppo)

    # Render pixel frames from both environments
    frame_dqn = env_dqn.render()
    frame_ppo = env_ppo.render()

    # Convert RGB frames to BGR format for OpenCV
    frame_dqn_bgr = cv2.cvtColor(frame_dqn, cv2.COLOR_RGB2BGR)
    frame_ppo_bgr = cv2.cvtColor(frame_ppo, cv2.COLOR_RGB2BGR)

    # Add text labels on top of each frame
    cv2.putText(
        frame_dqn_bgr,
        "DQN Agent",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
    )
    cv2.putText(
        frame_ppo_bgr,
        "PPO Agent",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 165, 255),
        2,
    )

    # Combine frames side-by-side horizontally
    combined_frame = np.hstack((frame_dqn_bgr, frame_ppo_bgr))

    video_writer.write(combined_frame)
    # Display combined video frame
    cv2.imshow("Lunar Lander Comparison: DQN (Left) vs PPO (Right)", combined_frame)

    # Break loop if user presses 'q' or 'Esc'
    key = cv2.waitKey(20) & 0xFF
    if key == ord("q") or key == 27:
        break

    # Reset environments if episode ends
    if term_dqn or trunc_dqn:
        obs_dqn, _ = env_dqn.reset()
    if term_ppo or trunc_ppo:
        obs_ppo, _ = env_ppo.reset()

env_dqn.close()
env_ppo.close()
video_writer.release()
cv2.destroyAllWindows()