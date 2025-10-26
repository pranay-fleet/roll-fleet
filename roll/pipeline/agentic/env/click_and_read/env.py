import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Any, Dict
import random
import logging

from gem import Env
from roll.pipeline.agentic.utils import all_seed

logger = logging.getLogger(__name__)


class ClickAndReadEnv(Env):
    """
    Simple VL + Tool Use Task:
    1. Agent sees 100x100 image with a 25x25 green button in one of 4 positions
    2. Agent must click the green button (correct coordinates)
    3. If clicked correctly, new image appears with a word
    4. Agent must read and submit the word
    5. Terminal reward: 1 if correct, 0 otherwise
    """

    def __init__(
        self,
        max_steps=5,
        image_size=100,
        button_size=25,
        format_penalty=-0.01,
        render_mode="rgb_array",
        **kwargs
    ):
        super().__init__()
        self.max_steps = max_steps
        self.image_size = image_size
        self.button_size = button_size
        self.format_penalty = format_penalty
        self.render_mode = render_mode

        # 4 fixed button positions (corners)
        margin = 0
        self.BUTTON_POSITIONS = {
            "top_left": (margin, margin),
            "top_right": (image_size - button_size - margin, margin),
            "bottom_left": (margin, image_size - button_size - margin),
            "bottom_right": (image_size - button_size - margin, image_size - button_size - margin),
        }
        
        # Word bank for the task
        self.WORDS = ["APPLE", "CHERRY", "EAGLE", "FOREST", "HELLO"]

        # State tracking
        self.current_step = 0
        self.button_clicked = False
        self.button_position = None  # (x, y) of top-left corner
        self.button_location_name = None  # "top_left", etc.
        self.target_word = None
        self.current_image = None

    def get_instructions(self) -> str:
        return (
            "You are an AI assistant that can see images and perform actions.\n\n"
            "Image: 100x100 pixels. Coordinate (0,0) is top-left corner, (99,99) is bottom-right corner.\n\n"
            "Available actions:\n"
            "1. <answer>click:[x,y]</answer> - Click at coordinates where 0 ≤ x,y ≤ 99\n"
            "2. <answer>submit:[word]</answer> - Submit your answer as a single word\n\n"
            "Task: Click the green button to reveal text, then submit the word you see."
        )

    def reset(self, seed=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        Env.reset(self, seed)
        
        # Reset state
        self.current_step = 0
        self.button_clicked = False
        
        with all_seed(seed):
            # Randomly select one of 4 positions
            self.button_location_name = random.choice(list(self.BUTTON_POSITIONS.keys()))
            self.button_position = self.BUTTON_POSITIONS[self.button_location_name]

            # Select random word
            self.target_word = random.choice(self.WORDS)
            
        logger.debug(
            f"Environment reset - Button: {self.button_location_name} {self.button_position}, "
            f"Target word: {self.target_word}"
        )
        
        # Render initial state (image with green button)
        self.current_image = self._render_button_image()
        
        return self.current_image, {"env_instruction": self.get_instructions()}

    def step(self, action: str) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        self.current_step += 1
        
        # Parse action
        action_info = self.parse_action(action)
        action_type = action_info["action_type"]
        action_is_valid = action_info["action_is_valid"]

        logger.debug(
            f"Step {self.current_step} - Action type: {action_type}, "
            f"Valid: {action_is_valid}, Info: {action_info}"
        )
        
        reward = 0.0
        terminated = False
        
        if not action_is_valid:
            # Format penalty for invalid action
            reward = self.format_penalty
            logger.debug(f"Invalid action format - penalty: {reward}")
        else:
            if action_type == "click":
                # Check if click is on the green button
                click_x, click_y = action_info["coordinates"]
                if self._is_click_on_button(click_x, click_y):
                    logger.debug(
                        f"Successful click at ({click_x}, {click_y}) - "
                        f"Button was at {self.button_position}"
                    )
                    self.button_clicked = True
                    self.current_image = self._render_word_image()
                else:
                    logger.debug(
                        f"Missed click at ({click_x}, {click_y}) - "
                        f"Button is at {self.button_position}"
                    )
                    # Image doesn't change
                    pass
                    
            elif action_type == "submit":
                # Check if the submitted word matches
                submitted_word = action_info["word"].upper().strip()
                if submitted_word == self.target_word and self.button_clicked:
                    reward = 1.0
                    logger.info(
                        f"SUCCESS! Submitted '{submitted_word}' matches target '{self.target_word}' "
                        f"and button was clicked"
                    )
                else:
                    reward = 0.0
                    logger.debug(
                        f"FAILURE - Submitted: '{submitted_word}', Target: '{self.target_word}', "
                        f"Button clicked: {self.button_clicked}"
                    )
                terminated = True
        
        # Check if max steps reached
        if self.current_step >= self.max_steps:
            if not terminated:
                # Didn't submit in time - no reward
                reward = 0.0
                logger.debug(f"Episode terminated - max steps ({self.max_steps}) reached")
            terminated = True
        
        truncated = (self.current_step >= self.max_steps) and not terminated
        
        # Metrics
        metrics = {
            "action_is_valid": action_is_valid,
            "button_clicked": self.button_clicked,
            "success": reward > 0,
            "steps_used": self.current_step,
        }
        
        info = {
            "metrics": metrics,
            "metrics_agg_mode": {
                "action_is_valid": "mean",
                "button_clicked": "last",
                "success": "last",
                "steps_used": "mean",
            },
            "button_position": self.button_location_name,  # Non-metric metadata
        }
        
        return self.current_image, reward, terminated, truncated, info

    def _is_click_on_button(self, x: int, y: int) -> bool:
        """Check if click coordinates are within the button bounds"""
        bx, by = self.button_position
        return (bx <= x <= bx + self.button_size and 
                by <= y <= by + self.button_size)

    def _render_button_image(self) -> np.ndarray:
        """Render image with green button at specified position"""
        img = Image.new('RGB', (self.image_size, self.image_size), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw green button
        bx, by = self.button_position
        draw.rectangle(
            [bx, by, bx + self.button_size, by + self.button_size],
            fill='green',
            outline='darkgreen',
            width=2
        )
        
        return np.array(img)

    def _render_word_image(self) -> np.ndarray:
        """Render image with the target word displayed"""
        img = Image.new('RGB', (self.image_size, self.image_size), color='lightblue')
        draw = ImageDraw.Draw(img)
        
        # Draw the word in large text
        try:
            font = ImageFont.load_default(size=25)
        except TypeError:
            # Older Pillow versions don't support size parameter
            font = ImageFont.load_default()
        
        # Center the word
        text_bbox = draw.textbbox((0, 0), self.target_word, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        text_x = (self.image_size - text_w) // 2
        text_y = (self.image_size - text_h) // 2
        
        draw.text((text_x, text_y), self.target_word, fill='black', font=font)
        
        return np.array(img)

    def parse_action(self, text: str) -> Dict:
        """Parse action from LLM output"""
        import re
        
        # Remove chat tokens
        for token in ("<|im_start|>", "<|im_end|>"):
            text = text.replace(token, "")
        
        # Pattern: <answer>click:[x,y]</answer> or <answer>submit:[word]</answer>
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        
        if not match:
            return {
                "action_type": None,
                "action_is_valid": False,
                "format_penalty": self.format_penalty,
            }
        
        content = match.group(1).strip()
        
        # Parse click action
        click_match = re.match(r"click:\s*\[?(\d+),\s*(\d+)\]?", content, re.IGNORECASE)
        if click_match:
            x, y = int(click_match.group(1)), int(click_match.group(2))
            return {
                "action_type": "click",
                "coordinates": (x, y),
                "action_is_valid": True,
                "format_penalty": 0.0,
            }
        
        # Parse submit action
        submit_match = re.match(r"submit:\s*(.+)", content, re.IGNORECASE)
        if submit_match:
            word = submit_match.group(1).strip()
            return {
                "action_type": "submit",
                "word": word,
                "action_is_valid": True,
                "format_penalty": 0.0,
            }
        
        # Invalid format
        return {
            "action_type": None,
            "action_is_valid": False,
            "format_penalty": self.format_penalty,
        }

